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


// Copmute gaussian blur per channel on the CPU.
// Call this function for each of red, green, and blue channels
// Returns blurred channel.
void ComputeConvolutionOnCPU(unsigned char* const blurredChannel, const unsigned char* const inputChannel, int rows, int cols, float* filter, int filterWidth)
{
    // Filter width should be odd as we are calculating average blur for a pixel plus some offset in all directions

    const int half = filterWidth / 2;
    const int width = cols - 1;
    const int height = rows - 1;

    // Compute blur
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
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
                    int             idx = w + cols * h;                                                                                 // current pixel index
                    float   pixel = static_cast<float>(inputChannel[idx]);

                    idx = (i + half) * filterWidth + j + half;
                    float   weight = filter[idx];

                    blur += pixel * weight;
                }
            }

            blurredChannel[c + cols * r] = static_cast<unsigned char>(blur);
        }
    }
}

void GaussianBlurOnCPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float* filter, int  filterWidth)
{
    const int numPixels = rows * cols;

    // Create channel variables
    unsigned char* red = new unsigned char[numPixels];
    unsigned char* green = new unsigned char[numPixels];
    unsigned char* blue = new unsigned char[numPixels];

    unsigned char* redBlurred = new unsigned char[numPixels];
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

    // Compute convolution for each individual channel



    ComputeConvolutionOnCPU(redBlurred, red, rows, cols, filter, filterWidth);
    ComputeConvolutionOnCPU(greenBlurred, green, rows, cols, filter, filterWidth);
    ComputeConvolutionOnCPU(blueBlurred, blue, rows, cols, filter, filterWidth);



    // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
    for (int p = 0; p < 100; ++p) //numPixels
    {
        unsigned char r = redBlurred[p];
        printf("%c ", r);
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

__global__ void  ComputeConvolutionOnGPU(unsigned char* blurredChannel, const unsigned char* const inputChannel,
    int *rows, int *cols, float* filter, int *filterWidth)
{

    int c = blockIdx.x * blockDim.x + threadIdx.x; //colun
    int r = blockIdx.y * blockDim.y + threadIdx.y; //row


    if ((*rows) > r && (*cols) > c) {
        //si esta dentro...

        const int half = (*filterWidth) / 2;
        const int width = (*cols) - 1;
        const int height = (*rows);

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

        blurredChannel[c + (*cols) * r] =  static_cast<unsigned char>(blur);

    }
}

void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float* filter, int  filterWidth,
    dim3 blockShape, int numBlocksFilas, int numBlocksColumnas, int blocksPorGrid)
{
    const int numPixels = rows * cols;

    // Create channel variables
    unsigned char* red = new unsigned char[numPixels];
    unsigned char* green = new unsigned char[numPixels];
    unsigned char* blue = new unsigned char[numPixels];

    unsigned char* redBlurred = new unsigned char[numPixels];
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

    // Compute convolution for each individual channel

    unsigned char* dev_red;// = new unsigned char[numPixels];
    unsigned char* dev_redBlurred;// = new unsigned char[numPixels];

    int gridSizePx = blockShape.y * cols * sizeof(unsigned char);
    int addRows = cols * sizeof(unsigned char) * (filterWidth / 2);
    int ultima = numBlocksColumnas - 1;
    int rowsGrid;


    int* dev_rowsGrid, * dev_cols, * dev_filterWidth;
    float* dev_filter;

    cudaMalloc(&dev_rowsGrid, sizeof(int));
    cudaMalloc(&dev_cols, sizeof(int));
    cudaMalloc(&dev_filterWidth, sizeof(int));
    cudaMalloc(&dev_filter, filterWidth * filterWidth * sizeof(float));


    cudaMemcpy(dev_cols, &cols, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filterWidth, &filterWidth, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter, &filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t cudaStatus;

    //primera iter

    cudaMalloc(&dev_red, numPixels * sizeof(unsigned char));
    cudaMalloc(&dev_redBlurred, numPixels * sizeof(unsigned char));

    cudaStream_t *streams1 = (cudaStream_t*) malloc(numBlocksColumnas * sizeof(cudaStream_t));


    printf("gridsizePx: %d \naddRows: %d \nultima: %d\nnumpixels: %d \nnumBlocksColumna: %d \nrows: %d \ncolumns: %d", gridSizePx, addRows, ultima,numPixels, numBlocksColumnas,rows,cols);
    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStreamCreate(&streams1[i]);
        rowsGrid = blockShape.y;

        if (i != ultima)
            cudaMemcpyAsync(dev_red + gridSizePx * i, red + gridSizePx * i, gridSizePx + addRows, cudaMemcpyHostToDevice, streams1[i]);
        else {
            cudaMemcpyAsync(dev_red + gridSizePx * i, red + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams1[i]);
            rowsGrid = rows % blockShape.y;
        }
    
        cudaMemcpyAsync(dev_rowsGrid, &rowsGrid, sizeof(int), cudaMemcpyHostToDevice, streams1[i]);
        ComputeConvolutionOnGPU << <numBlocksFila, blockShape, 0, streams1[i] >> > (dev_redBlurred + i * gridSizePx,
            dev_red + i * gridSizePx, dev_rowsGrid, dev_cols, dev_filter, dev_filterWidth);

        //printf("%d", i);

       /* cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after functino!\n", cudaStatus);
            return;
        }*/
    }


    unsigned char* dev_green; //= new unsigned char[numPixels];
    unsigned char* dev_greenBlurred; //= new unsigned char[numPixels];

    cudaMalloc(&dev_green, numPixels * sizeof(unsigned char));
    cudaMalloc(&dev_greenBlurred, numPixels * sizeof(unsigned char));
    cudaStream_t* streams2 = (cudaStream_t*)malloc(numBlocksColumnas * sizeof(cudaStream_t));


    //ComputeConvolutionOnGPU(greenBlurred, green, rows, cols, filter, filterWidth);
    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStreamCreate(&streams2[i]);
         rowsGrid = blockShape.y;

        if (i != ultima)
            cudaMemcpyAsync(dev_green + gridSizePx * i, green + gridSizePx * i, gridSizePx + addRows, cudaMemcpyHostToDevice, streams2[i]);
        else {
            cudaMemcpyAsync(dev_green + gridSizePx * i, green + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams2[i]);
            rowsGrid = rows % blockShape.y;
        }
           

    //    ComputeConvolutionOnGPU <<<numBlocksFila, blockShape, 0, streams2[i] >>> (dev_greenBlurred + i * gridSizePx,
      //      dev_green + i * gridSizePx, blockShape.x, cols, filter, filterWidth);
        cudaMemcpyAsync(dev_rowsGrid, &rowsGrid, sizeof(int), cudaMemcpyHostToDevice, streams2[i]);
        ComputeConvolutionOnGPU << <numBlocksFila, blockShape, 0, streams2[i] >> > (dev_greenBlurred + i * gridSizePx,
            dev_green + i * gridSizePx, dev_rowsGrid, dev_cols, dev_filter, dev_filterWidth);
    }


    unsigned char* dev_blue; //= new unsigned char[numPixels];
    unsigned char* dev_blueBlurred; // = new unsigned char[numPixels];

    cudaMalloc(&dev_blue, numPixels * sizeof(unsigned char));
    cudaMalloc(&dev_blueBlurred, numPixels * sizeof(unsigned char));

    cudaStream_t* streams3 = (cudaStream_t*)malloc(numBlocksColumnas * sizeof(cudaStream_t));

    //ComputeConvolutionOnGPU(blueBlurred, blue, rows, cols, filter, filterWidth);
    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStreamCreate(&streams3[i]);
        rowsGrid = blockShape.y;
        if (i != ultima)
            cudaMemcpyAsync(dev_blue + gridSizePx * i, blue + gridSizePx * i, gridSizePx + addRows, cudaMemcpyHostToDevice, streams3[i]);
        else {
            cudaMemcpyAsync(dev_blue + gridSizePx * i, blue + gridSizePx * i, gridSizePx, cudaMemcpyHostToDevice, streams3[i]);
            rowsGrid = rows % blockShape.y;
        }

        //ComputeConvolutionOnGPU << <numBlocksFila, blockShape, 0, streams3[i] >> > (dev_blueBlurred + i * gridSizePx,
          //  dev_blue + i * gridSizePx, blockShape.x, cols, filter, filterWidth);

        cudaMemcpyAsync(dev_rowsGrid, &rowsGrid, sizeof(int), cudaMemcpyHostToDevice, streams3[i]);
        ComputeConvolutionOnGPU << <numBlocksFila, blockShape, 0, streams3[i] >> > (dev_blueBlurred + i * gridSizePx,
            dev_blue + i * gridSizePx, dev_rowsGrid, dev_cols, dev_filter, dev_filterWidth);
    }


    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStatus =  cudaStreamSynchronize(streams1[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d stream1!\n", cudaStatus);
            return;
        }
        cudaStreamDestroy(streams1[i]);
    }

    //Se hace la primera copia de resultados
    cudaMemcpyAsync(redBlurred,dev_redBlurred, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStatus = cudaStreamSynchronize(streams2[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d stream1!\n", cudaStatus);
            return;
        }
        cudaStreamDestroy(streams2[i]);
    }

    //Se hace la segunda copia de resultados
    cudaMemcpyAsync(greenBlurred, dev_greenBlurred, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numBlocksColumnas; i++) {
        cudaStatus = cudaStreamSynchronize(streams3[i]);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d stream1!\n", cudaStatus);
            return;
        }
        cudaStreamDestroy(streams3[i]);
    }

    //Se hace la tercera copia de resultados
    cudaMemcpyAsync(blueBlurred, dev_blueBlurred, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaStatus = cudaDeviceSynchronize(); //fin
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return;
    }

    // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
    for (int p = 0; p < 100; ++p) //numPixels
    {
        unsigned char r = redBlurred[p];
        printf("%c ", r);
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
    originalImage = (uchar4*)imgRGBA.ptr<unsigned char>(0);;
    blurredImage = (uchar4*)malloc(width * height * sizeof(uchar4));

    //Tu práctica empieza aquí
    //CUDA	

    dim3 blockShape = dim3(8, 8);
    int numBlocksFilas = height / 8 + 1;
    int numBlocksColumnas = width / 8 + 1;
    int blocksPorGrid = numBlocksFilas;

    GaussianBlurOnGPU(blurredImage, originalImage, height, width, filter, filterWidth, blockShape,
        numBlocksFilas, numBlocksColumnas, blocksPorGrid);

    //Version CPU (Comentar cuando se trabaje con la GPU!)
   // GaussianBlurOnCPU(blurredImage, originalImage, height, width, filter, filterWidth);

    Mat out(height, width, CV_8UC4, blurredImage);

    imwrite(outputPath, out);

    printf("Done!\n");
    return 0;
}
