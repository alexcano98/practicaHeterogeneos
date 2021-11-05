#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

float* createFilter(int width)
{
    const float sigma = 2.f;                          // Standard deviation of the Gaussian distribution.

    const int       half = width / 2;
    float           sum = 0.f;


    // Create convolution matrix
    float* res = (float*)malloc(width * width * sizeof(float)); //int to long?


    // Calculate filter sum first
    for (int r = -half; r <= half; ++r)
    {
        for (int c = -half; c <= half; ++c)
        {
            // e (natural logarithm base) to the power x, where x is what's in the brackets
            float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
            int idx = (r + half) * width + c + half;

            res[idx] = weight;
            sum += weight;
        }
    }

    // Normalize weight: sum of weights must equal 1
    float normal = 1.f / sum;

    for (int r = -half; r <= half; ++r)
    {
        for (int c = -half; c <= half; ++c)
        {
            int idx = (r + half) * width + c + half;

            res[idx] *= normal;
        }
    }
    return res;
}


__global__ void  ComputeConvolutionOnGPU(unsigned char* blurredChannel1, unsigned char* blurredChannel2, unsigned char* blurredChannel3, const unsigned char* const inputChannel1, 
    const unsigned char* const inputChannel2,const unsigned char* const inputChannel3,int *rows, int *cols, float* filter, int *filterWidth)
{   
    int c = blockIdx.x * blockDim.x + threadIdx.x; //colun
    int r = blockIdx.y * blockDim.y + threadIdx.y; //row
    

    extern __shared__ float sh_filter[81];
    int proyeccion = threadIdx.x + blockDim.x * threadIdx.y;
    if (proyeccion < 81) {
        sh_filter[proyeccion] = filter[proyeccion];
    }

    if ( (*rows) > r && (*cols) > c) {
        //si esta dentro...

        __syncthreads(); //barrera para esperar que el filtro este copiado.    

        const int half = (*filterWidth) / 2;
        const int width = (*cols) -1;
        const int height = (*rows) -1;

       float blur1 = 0.f;
       float blur2 = 0.f;
       float blur3 = 0.f;
        // Average pixel color summing up adjacent pixels.
       for (int i = -half; i <= half; ++i)
        {
            for (int j = -half; j <= half; ++j)
            {
                // Clamp filter to the image border
                int h = min(max(r + i, 0), height);
                int w = min(max(c + j, 0), width);

                // Blur is a product of current pixel value and weight of that pixel.
                // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
                int       idx = w + (*cols) * h;                           // current pixel index
                float   pixel1 = static_cast<float>(inputChannel1[idx]);
                float   pixel2 = static_cast<float>(inputChannel2[idx]);
                float   pixel3 = static_cast<float>(inputChannel3[idx]);

                idx = (i + half) * (*filterWidth) + j + half;
                float   weight = sh_filter[idx];

                blur1 += pixel1 * weight;
                blur2 += pixel2 * weight;
                blur3 += pixel3 * weight;
            }
        }

       blurredChannel1[c + (*cols) * r] = static_cast<unsigned char>(blur1); //'a'; inputChannel[c + cols * r];
       blurredChannel2[c + (*cols) * r] = static_cast<unsigned char>(blur2); //'a'; inputChannel[c + cols * r];
       blurredChannel3[c + (*cols) * r] = static_cast<unsigned char>(blur3); //'a'; inputChannel[c + cols * r];

    }
}

void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda returned error code %d !\n", cudaStatus);
        printf("Error status is %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
}

void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float* filter, int  filterWidth)
{

    const int numPixels = rows * cols;
    dim3 blockShape = dim3(16, 16);
    int dim1 = (cols / blockShape.x) + 1;
    int dim2 = (rows / blockShape.y) + 1;
    dim3 gridShape = dim3( dim1 , dim2);


    checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );


    // Create channel variables
    unsigned char* red; // = new unsigned char[numPixels];
    checkCudaErrors(cudaMallocManaged(&red, numPixels)); //cudaMallocManaged ??

    unsigned char* green; // = new unsigned char[numPixels];
    checkCudaErrors(cudaMallocManaged(&green, numPixels));

    unsigned char* blue; // = new unsigned char[numPixels];
    checkCudaErrors(cudaMallocManaged(&blue, numPixels));
   
    
    unsigned char* redBlurred; // = new unsigned char[numPixels];  //(float *) malloc(numPixels * sizeof(float));
    checkCudaErrors(cudaMallocManaged(&redBlurred, numPixels));
    unsigned char* greenBlurred; // = new unsigned char[numPixels];
    checkCudaErrors(cudaMallocManaged(&greenBlurred, numPixels));
    unsigned char* blueBlurred; // = new unsigned char[numPixels];
    checkCudaErrors(cudaMallocManaged(&blueBlurred, numPixels));



    // Separate RGBAimage into red, green, and blue components
    for (int p = 0; p < numPixels; ++p)
    {
        uchar4 pixel = rgba[p];

        red[p] = pixel.x;
        green[p] = pixel.y;
        blue[p] = pixel.z;
    }
    printf("paso");
    //Init commom cuda variables

    int *dev_rows,  *dev_cols,  *dev_filterWidth;
    float* dev_filter;

    // ¿La memoria unificada es coherente? ¿Vale la pena?

    checkCudaErrors( cudaMalloc(&dev_rows, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_cols, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_filterWidth, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_filter, (filterWidth) * (filterWidth) * sizeof(float)) );
    

    checkCudaErrors( cudaMemcpy(dev_cols, &cols, sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_rows, &rows, sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_filterWidth, &filterWidth, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors( cudaMemcpy(dev_filter, filter, (filterWidth) * (filterWidth) * sizeof(float), cudaMemcpyHostToDevice) );   


    // Compute convolution for each individual channel
    cudaStream_t streams[3];

    //primera iter
    cudaStreamCreate(&streams[0]);
    checkCudaErrors(cudaMemPrefetchAsync(redBlurred, numPixels, 0));
    checkCudaErrors(cudaMemPrefetchAsync(red, numPixels, 0));
    
    //segunda iter
    cudaStreamCreate(&streams[1]);
    checkCudaErrors(cudaMemPrefetchAsync(greenBlurred, numPixels, 0));
    checkCudaErrors(cudaMemPrefetchAsync(green, numPixels, 0));

    //tercera iter
    cudaStreamCreate(&streams[2]);
    checkCudaErrors(cudaMemPrefetchAsync(blueBlurred, numPixels, 0));
    checkCudaErrors(cudaMemPrefetchAsync(blue, numPixels, 0));
    
    ComputeConvolutionOnGPU << <gridShape, blockShape,0, streams[2] >> > (redBlurred, greenBlurred, blueBlurred, red, green, blue, dev_rows, dev_cols, dev_filter, dev_filterWidth);

    checkCudaErrors(cudaMemPrefetchAsync(redBlurred, numPixels, cudaCpuDeviceId));
    checkCudaErrors(cudaMemPrefetchAsync(greenBlurred, numPixels, cudaCpuDeviceId));
    checkCudaErrors(cudaMemPrefetchAsync(blueBlurred, numPixels, cudaCpuDeviceId));

    checkCudaErrors(cudaDeviceSynchronize()); //fin

    
    
    

    // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
    for (int p = 0; p < numPixels; ++p) //numPixels
    {
        unsigned char r = redBlurred[p];
        //printf("%c ", redBlurred[p]);
        unsigned char g = greenBlurred[p];
        unsigned char b = blueBlurred[p];

        modifiedImage[p] = make_uchar4(r, g, b, 255);
    }

    
    /*delete[] red;
    delete[] green;
    delete[] blue;
    delete[] redBlurred;
    delete[] greenBlurred;
    delete[] blueBlurred;*/

   
    cudaFree(red);
    cudaFree(green);
    cudaFree(blue);
    cudaFree(redBlurred);
    cudaFree(greenBlurred);
    cudaFree(blueBlurred);

}


Mat getRandomImage() {

    Mat img = Mat::ones(20000, 20000, CV_8UC3) *4;
    return img;
}


// Main entry into the application
int main(int argc, char** argv)
{
    char* imagePath;
    char* outputPath;

    int height, width, channels;
    uchar4* originalImage, * blurredImage;

    int filterWidth = 9;
    float* filter = createFilter(filterWidth);

    if (argc > 2)
    {
        imagePath = argv[1];
        outputPath = argv[2];
    }
    else
    {
        printf("Please provide input and output image files as arguments to this application.");
        exit(1);
    }

    //printf("imagen %s \n", imagePath);

    Mat img;
    Mat imgRGBA;
  
    //Read the image
    img = imread(imagePath, IMREAD_COLOR); //problemas para muchos pixeles

    if (img.empty()) printf("Could not load image file: %s\n", imagePath);
    
    //img = getRandomImage(); //lo pongo random por el tamaño

    // get the image data
    height = img.rows;
    width = img.cols;
    channels = img.channels();

    cvtColor(img, imgRGBA, COLOR_BGR2BGRA);

    //Allocate and copy
    originalImage = (uchar4*) imgRGBA.ptr<unsigned char>(0);;
    blurredImage = (uchar4*) malloc(width * height * sizeof(uchar4));

    //Tu práctica empieza aquí
    //CUDA	

    GaussianBlurOnGPU(blurredImage, originalImage, height, width, filter, filterWidth);

    //Version CPU (Comentar cuando se trabaje con la GPU!)
   // GaussianBlurOnCPU(blurredImage, originalImage, height, width, filter, filterWidth);

    Mat out(height, width, CV_8UC4, blurredImage);

    imwrite(outputPath, out);

    printf("Done!\n");
    return 0;
}



