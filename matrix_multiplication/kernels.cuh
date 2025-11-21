#ifndef TILE
#define TILE 32
#endif

#ifndef SUBTILE
#define SUBTILE 64
#endif

#ifndef SUB
#define SUB 4
#endif


// GEMM with baseline kernel
__global__ void GEMMBaseline(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,  // (M x K)
                           const float* __restrict__ B,  // (K x N)
                           float beta,
                           float* __restrict__ C);  // (M x N))

// GEMM with Tiling
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,  // (M x K)
                           const float* __restrict__ B,  // (K x N)
                           float beta,
                           float* __restrict__ C);



// GEMM with SubTiling
template <typename LoaderFunc, typename ComputeFunc>
__global__ void GEMMSubTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,  // (M x K)
                           const float* __restrict__ B,  // (K x N)
                           float beta,
                           float* __restrict__ C)

{
    __shared__ float ATile[SUBTILE][SUBTILE+1]; // Padding to avoid bank conflicts
    __shared__ float BTile[SUBTILE][SUBTILE+1];



    // Grid-wide stride
    int strideRow = SUBTILE * gridDim.y;
    int strideCol = SUBTILE * gridDim.x;

    //Thread row and column within origin within the tile
    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;


    //Loop over all origin rows and columns that a given block is responsible for
    for (int startRow = blockIdx.y * SUBTILE; startRow < M; startRow += strideRow) {
        for (int startCol = blockIdx.x * SUBTILE; startCol < N; startCol += strideCol){

            int threadRowGlobalOrigin = startRow + threadRowTile;
            int threadColGlobalOrigin = startCol + threadColTile;

            float sum[SUB][SUB];
            
            #pragma unroll
            for (int i = 0; i < SUB; i++){
                #pragma unroll
                for (int j = 0; j < SUB; j++){
                    sum[i][j] = 0.0f;
                }
            }


            for (int chunk = 0; chunk < K; chunk += SUBTILE){
                int threadRowGlobalOriginA = threadRowGlobalOrigin;
                int threadColGlobalOriginA = threadColTile + chunk;

                int threadRowGlobalOriginB = threadRowTile + chunk;
                int threadColGlobalOriginB = threadColGlobalOrigin;
                //First load into shared memory

                LoaderFunc::run(A, ATile, B, BTile, M, K, N,
                                startRow,  startCol, chunk,
                                threadRowTile, threadColTile,
                                threadRowGlobalOriginA, threadColGlobalOriginA,
                                threadRowGlobalOriginB, threadColGlobalOriginB);
                 __syncthreads();

                
                // Continue accumulation
                int kmax = min(SUBTILE, K-chunk);
                ComputeFunc::run(ATile, BTile, K, kmax,
                                sum, threadRowTile, threadColTile);

       
                __syncthreads();
            }
        //Write back to memory
            #pragma unroll
            for (int i = 0; i < SUB; i++){
                int threadRowGlobal = threadRowGlobalOrigin + i;
                if (threadRowGlobal >= M) break;
                #pragma unroll
                for (int j = 0; j < SUB; j++){
                    int threadColGlobal = threadColGlobalOrigin + j;
                    if (threadColGlobal >= N) break;
                        int idx = threadRowGlobal * N + threadColGlobal;
                        float COld = (beta != 0.0f) ? C[idx] : 0.0f;
                        C[idx] = alpha * sum[i][j] + beta *COld;
                    
                }
       
            }
        }
    }
}







__device__ __forceinline__
void load_subtile_naive(const float* __restrict__ A,
                        float ATile[SUBTILE][SUBTILE+1],
                        const float* __restrict__ B,
                        float BTile[SUBTILE][SUBTILE+1],
                        int M, int K, int N,
                        int startRow, int startCol, int chunk,
                        int threadRowTile, int threadColTile,
                        int threadRowGlobalOriginA, int threadColGlobalOriginA,
                        int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;

        #pragma unroll
        for (int j = 0; j < SUB; j++){

            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTile = threadColTile + j;
        

            //Loads into shared memory, in 
            ATile[rowTile][colTile] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
            BTile[rowTile][colTile] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
        }
    }
}


// Vectorized version (float4 loads)

__device__ __forceinline__
void load_subtile_vec4(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol, int chunk,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;
        #pragma unroll
        for (int j = 0; j < SUB; j+=4){

            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTile = threadColTile + j;

            //Prefer the floa4 load when possible
            int idxA = rowA * K + colA;
            if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = reinterpret_cast<const float4*>(&A[rowA * K + colA])[0];
                ATile[rowTile][colTile + 0] = tmpA.x;
                ATile[rowTile][colTile + 1] = tmpA.y;
                ATile[rowTile][colTile + 2] = tmpA.z;
                ATile[rowTile][colTile + 3] = tmpA.w;
            } 
            //Otherwise fall back to scalar loads
            else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colAGlobal = colA + k;
                    ATile[rowTile][colTile + k] =
                        (rowA < M && colAGlobal < K) ? A[rowA * K + colAGlobal] : 0.0f;
                }
            }
            int idxB = rowB * N + colB;
            if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = reinterpret_cast<const float4*>(&B[rowB * N +  colB])[0];
                BTile[rowTile][colTile + 0] = tmpB.x;
                BTile[rowTile][colTile + 1] = tmpB.y;
                BTile[rowTile][colTile + 2] = tmpB.z;
                BTile[rowTile][colTile + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colBGlobal = colB + k;
                    BTile[rowTile][colTile + k] =
                        (rowB < K && colBGlobal < N) ? B[rowB * N + colBGlobal] : 0.0f;
                }
            }
    
        }
    }
}


// Vectorized + swizzle to avoid bank conflicts in B
__device__ __forceinline__
void load_subtile_vec4_swizzle(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol, int chunk,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB)

{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;
        int colSwz  = (rowTile & 1) << 1;
        

        #pragma unroll
        for (int j = 0; j < SUB; j+=4){

            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTileA = threadColTile + j; 
            

            
            int colTileB = threadColTile + j;// only change from previous version

            int colTileBSwz = colTileB ^ colSwz;
            //Prefer the floa4 load when possible
            int idxA = rowA * K + colA;
            if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = reinterpret_cast<const float4*>(&A[rowA * K + colA])[0];

                ATile[rowTile][colTileA + 0] = tmpA.x;
                ATile[rowTile][colTileA + 1] = tmpA.y;
                ATile[rowTile][colTileA + 2] = tmpA.z;
                ATile[rowTile][colTileA + 3] = tmpA.w;
            } 
            //Otherwise fall back to scalar loads
            else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colAGlobal = colA + k;
                    ATile[rowTile][colTileA + k] =
                        (rowA < M && colAGlobal < K) ? A[rowA * K + colAGlobal] : 0.0f;
                }
            }
            int idxB = rowB * N + colB;
            if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = reinterpret_cast<const float4*>(&B[rowB * N +  colB])[0];
                BTile[rowTile][colTileBSwz + 0] = tmpB.x;
                BTile[rowTile][colTileBSwz + 1] = tmpB.y;
                BTile[rowTile][colTileBSwz + 2] = tmpB.z;
                BTile[rowTile][colTileBSwz + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colBGlobal = colB + k;
                    BTile[rowTile][colTileBSwz + k] =
                        (rowB < K && colBGlobal < N) ? B[rowB * N + colBGlobal] : 0.0f;
                }
            }
    
        }
    }
}

__device__ __forceinline__
void compute_subtile_naive(const float ATile[SUBTILE][SUBTILE+1],
                     const float BTile[SUBTILE][SUBTILE+1],
                     int K, int kmax,
                     float sum[SUB][SUB], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATile[threadRowTile + i][k];
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][threadColTile + j];
        }

        #pragma unroll
        for (int i = 0; i < SUB; i++){
            #pragma unroll
            for (int j = 0; j < SUB; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

        }

    }
}


__device__ __forceinline__
void compute_subtile_swizzle(const float ATile[SUBTILE][SUBTILE+1],
                     const float BTile[SUBTILE][SUBTILE+1],
                     int K, int kmax,
                     float sum[SUB][SUB], int threadRowTile, int threadColTile)
{   
    
            
    #pragma unroll
    for (int k = 0; k < kmax; k++){ 
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATile[threadRowTile + i][k];
        }
        int colSwz = (k & 1)<<1; 
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][(threadColTile + j) ^ colSwz];
        }

        #pragma unroll
        for (int i = 0; i < SUB; i++){
            #pragma unroll
            for (int j = 0; j < SUB; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

        }

    }
}


// __device__ __forceinline__
// void compute_subtile_swizzle_warp(const float ATile[SUBTILE][SUBTILE+1],
//                      const float BTile[SUBTILE][SUBTILE+1],
//                      int K, int kmax,
//                      float sum[SUB][SUB], int threadRowTile, int threadColTile)

// {

// }


struct LoaderNaive {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_naive(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderVec4 {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_vec4(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderVec4Swizzle {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_vec4_swizzle(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct ComputeNaive {
    __device__ __forceinline__
    static void run(
        const float ATile[SUBTILE][SUBTILE+1],
        const float BTile[SUBTILE][SUBTILE+1],
        int K, int kmax,
        float sum[SUB][SUB],
        int threadRowTile, int threadColTile
    ) {
        compute_subtile_naive(
            ATile, BTile,
            K,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
    }
};


struct ComputeSwizzle {
    __device__ __forceinline__
    static void run(
        const float ATile[SUBTILE][SUBTILE+1],
        const float BTile[SUBTILE][SUBTILE+1],
        int K, int kmax,
        float sum[SUB][SUB],
        int threadRowTile, int threadColTile
    ) {
        compute_subtile_swizzle(
            ATile, BTile,
            K, kmax,
            sum,
            threadRowTile, threadColTile
        );
    }
};
