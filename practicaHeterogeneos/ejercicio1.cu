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


__global__ void  ComputeConvolutionOnGPU(unsigned char* blurredChannel, const unsigned char* const inputChannel,
    int *rows, int *cols, float* filter, int *filterWidth)
{   
    int c = blockIdx.x * blockDim.x + threadIdx.x; //colun
    int r = blockIdx.y * blockDim.y + threadIdx.y; //row


    if ( (*rows) > r && (*cols) > c) {
        //si esta dentro...

        const int half = (*filterWidth) / 2;
        const int width = (*cols) -1;
        const int height = (*rows) -1;

       float blur = 0.f;
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
                float   pixel = static_cast<float>(inputChannel[idx]);

                idx = (i + half) * (*filterWidth) + j + half;
                float   weight = filter[idx];

                blur += pixel * weight;
            }
        }

       blurredChannel[c + (*cols) * r] = static_cast<unsigned char>(blur); //'a'; inputChannel[c + cols * r];

    }
}

void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda returned error code %d !\n", cudaStatus);
        return;
    }
}

void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float* filter, int  filterWidth)
{

    const int numPixels = rows * cols;
    dim3 blockShape = dim3(32, 32);
    int dim1 = (cols / blockShape.x) + 1;
    int dim2 = (rows / blockShape.y) + 1;
    dim3 gridShape = dim3( dim1 , dim2);

    // Create channel variables
    unsigned char* red = new unsigned char[numPixels];
    unsigned char* green = new unsigned char[numPixels];
    unsigned char* blue = new unsigned char[numPixels];

    unsigned char* redBlurred = new unsigned char[numPixels];  //(float *) malloc(numPixels * sizeof(float));
    unsigned char* greenBlurred = new unsigned char[numPixels];
    unsigned char* blueBlurred = new unsigned char[numPixels];


    // Separate RGBAimage into red, green, and blue components
    for (int p = 0; p < numPixels; ++p)
    {
        uchar4 pixel = rgba[p];

        red[p] = pixel.x;
        green[p] = pixel.y;
        blue[p] = pixel.z;
    }

    //Init commom cuda variables

    int *dev_rows,  *dev_cols,  *dev_filterWidth;
    float* dev_filter;


    checkCudaErrors( cudaMalloc(&dev_rows, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_cols, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_filterWidth, sizeof(int)) );
    checkCudaErrors( cudaMalloc(&dev_filter, (filterWidth) * (filterWidth) * sizeof(float)) );


    checkCudaErrors( cudaMemcpy(dev_cols, &cols, sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_rows, &rows, sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(dev_filterWidth, &filterWidth, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors( cudaMemcpy(dev_filter, filter, (filterWidth) * (filterWidth) * sizeof(float), cudaMemcpyHostToDevice) ); //creo q estaba mnal aqui


    //cudaStream_t* streams = (cudaStream_t*)malloc(3 * sizeof(cudaStream_t));
   


    // Compute convolution for each individual channel

    //primera iter
    unsigned char* dev_red;
    unsigned char* dev_redBlurred;

    //checkCudaErrors( cudaStreamCreate(&streams[0]) );

    cudaMalloc(&dev_red, numPixels );
    cudaMalloc(&dev_redBlurred, numPixels );

    cudaMemcpy(dev_red, red, numPixels, cudaMemcpyHostToDevice);

    ComputeConvolutionOnGPU <<<gridShape,blockShape>>>(dev_redBlurred, dev_red, dev_rows, dev_cols, dev_filter, dev_filterWidth);
    //ComputeConvolutionOnGPU(float* blurredChannel, const unsigned char* const inputChannel, int* rows, int* cols, float* filter, int* filterWidth)
    

    //segunda iter
    unsigned char* dev_green;
    unsigned char* dev_greenBlurred;

    //checkCudaErrors( cudaStreamCreate(&streams[1]) );

    cudaMalloc(&dev_green, numPixels);
    cudaMalloc(&dev_greenBlurred, numPixels);
 
    cudaMemcpy(dev_green, green, numPixels, cudaMemcpyHostToDevice);

    ComputeConvolutionOnGPU << <gridShape, blockShape >> > (dev_greenBlurred , dev_green , dev_rows, dev_cols, dev_filter, dev_filterWidth);


    //tercera iter
    unsigned char* dev_blue; 
    unsigned char* dev_blueBlurred;

    cudaMalloc(&dev_blue, numPixels);
    cudaMalloc(&dev_blueBlurred, numPixels);

    //checkCudaErrors( cudaStreamCreate(&streams[2]) );
    cudaMemcpy(dev_blue, blue, numPixels, cudaMemcpyHostToDevice);

    ComputeConvolutionOnGPU << <gridShape, blockShape>> > (dev_blueBlurred, dev_blue, dev_rows, dev_cols, dev_filter, dev_filterWidth);

    checkCudaErrors(cudaDeviceSynchronize()); //fin

   /*checkCudaErrors(cudaStreamSynchronize(streams[0]));
    checkCudaErrors( cudaStreamDestroy(streams[0]) );*/
    

    //Se hace la primera copia de resultados
    checkCudaErrors( cudaMemcpy(redBlurred, dev_redBlurred, numPixels, cudaMemcpyDeviceToHost) );


    /*checkCudaErrors( cudaStreamSynchronize(streams[1]) );
    checkCudaErrors( cudaStreamDestroy(streams[1]) );*/


    //Se hace la segunda copia de resultados
    checkCudaErrors( cudaMemcpy(greenBlurred, dev_greenBlurred, numPixels, cudaMemcpyDeviceToHost) );

    
    /*checkCudaErrors(cudaStreamSynchronize(streams[2]));
    checkCudaErrors( cudaStreamDestroy(streams[2]) );*/
    

    //Se hace la tercera copia de resultados
    checkCudaErrors( cudaMemcpy(blueBlurred, dev_blueBlurred, numPixels, cudaMemcpyDeviceToHost) );


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

    delete[] red;
    delete[] green;
    delete[] blue;
    delete[] redBlurred;
    delete[] greenBlurred;
    delete[] blueBlurred;
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


    Mat img;
    Mat imgRGBA;

    //Read the image
    img = imread(imagePath, IMREAD_COLOR);
    if (img.empty()) printf("Could not load image file: %s\n", imagePath);

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
