// pds-to-avi-v0.0.1.cpp

// opencv api
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

// pixelink API
#include <PixeLINKApi.h>

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <direct.h>
#include <sstream>
#include <string>
#include <assert.h>
#include <cmath>
#include <sys/stat.h>
// #include <boost/program_options.hpp>

// using namespace boost;
using namespace std;
using namespace cv;

// reference for input parameters
enum InputParams {
    PARAM_EXE_NAME = 0,
    PARAM_PDS_FILE_NAME,
    MIN_NUM_PARAMS
};

// function prototypes
static void            help();
static PXL_RETURN_CODE ConvertPdsToAvi(const char* pFileName);
static PXL_RETURN_CODE ConvertPdsToAviHelper(
    FILE* hFile, 
    char const* const pOutputFilePath); 
static float		   GetPixelSize(U32 pixelFormat);

// main program
int main(int argCount, char* argv[]) {

    // program_options::

    if (argCount < MIN_NUM_PARAMS) {
        printf("Too few arguments passed, only got %d, expected %d.\n",
            (argCount - 1),
            ((int)MIN_NUM_PARAMS - 1));
        help();
        return -1;
    }

    PXL_RETURN_CODE returnCode = ConvertPdsToAvi(argv[PARAM_PDS_FILE_NAME]); 

    if (!API_SUCCESS(returnCode)) {
        printf("An error occured when attempting conversion of pds file"
            "\n\t\"%s\"\n\tError Code: (0x%8.8X)\n",
            argv[PARAM_PDS_FILE_NAME],
            returnCode); 
        return EXIT_FAILURE;
    }

    printf("SUCCESS: Exiting program.\n");
    return EXIT_SUCCESS;
}


// help & usage message to console
static void 
help() {
    printf("Usage:\n\tpdv-to-avi <pds_file_name>\n");
}

// main wrapper function for converting pds to AVI
PXL_RETURN_CODE ConvertPdsToAvi(const char* pFileName)
{
    printf("Beginning conversion of PDS to avi file.\n");
    printf("Checking if file exists: \n\t\"%s\"\n", pFileName);

    if (NULL == pFileName) {
        printf("ERROR: a valid filename was not passed (null pointer recieved).\n");
        return -1;
    }
    // open file for binary read
    FILE* hFile;
    errno_t fOpenErr = fopen_s(&hFile, pFileName, "rb");
    if ((fOpenErr != 0) || (NULL == hFile)) {
        printf("IOERROR: PDS file could not be sucessfully opened, it may not exist.\n");
        return ApiInvalidParameterError;
    }
    printf("File exists and has been opened for binary read.\n");

    // generate timestamp
    auto rawTime = time(nullptr);
    struct tm timeInfo;
    localtime_s(&timeInfo, &rawTime); 
    stringstream transTime;
    transTime << put_time(&timeInfo, "%Y%m%d-%H%M%S");
    string timestamp = transTime.str();
    
    // handle paths   
    string inputFilePath(pFileName);
    size_t inputDirPos = inputFilePath.find_last_of("\\");
    string inputFileDir = inputFilePath.substr(0, inputDirPos); 
    string inputFileName = inputFilePath.substr(
        inputDirPos, inputFilePath.length());
    size_t nameOnlyPos = inputFileName.find_last_of(".");    
    char outputFileSubDir[MAX_PATH];
    
    // TODO - setup conversion optaion to add time stamp
    // sprintf_s(&outputFileSubDir[0],  
    //     sizeof(outputFileSubDir),
    //     "%s_CONVERTED-%s", 
    //     inputFileName.substr(0, nameOnlyPos).c_str(),
    //     timestamp.c_str());

    sprintf_s(&outputFileSubDir[0],
        sizeof(outputFileSubDir),
        "%s_CONVERTED",
        inputFileName.substr(0, nameOnlyPos).c_str());
    string outputFileName = inputFileName.substr(0, nameOnlyPos) + ".avi";
    string outputFileDir = inputFileDir + outputFileSubDir; 
    string outputFilePath = outputFileDir + outputFileName; 
    printf("Output File Path:\n\t\"%s\"\n", outputFilePath.c_str());

    struct stat dirInfo;

    int mkdirErr = _mkdir(outputFileDir.c_str());
    if (mkdirErr == ENOENT) {
        printf("ERROR: Could not create output directory.\n");
        return -1;
    } 
    else if (mkdirErr == EEXIST) {
        printf("ERROR: Could not create output directory "
            "(it is possible that a file exists with same name).\n");
        return -1;
    }

    // run main conversion implementation

    PXL_RETURN_CODE returnCode = ConvertPdsToAviHelper(hFile, outputFilePath.c_str());  
    
    // handle errors after run
    if (!API_SUCCESS(returnCode)) {        
        printf("ERROR: Conversion failed.\n");
    }

    // close binary PDS file
    printf("Closing PDS file.\n");
    fclose(hFile); 

    return returnCode; 
}

PXL_RETURN_CODE ConvertPdsToAviHelper(FILE* hFile, char const * const pOutputFilePath)
{
   
    PXL_RETURN_CODE returnCode; 

    // magic number validating the pds file
    U32 magicNumber = 0;
    // read the fist 4 bytes to make sure this is a valid pds file
    size_t numItemsRead = fread(
        (void*)&magicNumber, 
        sizeof(magicNumber), 
        1, 
        hFile);
    if (numItemsRead != 1) {
        printf("IOERROR: Failed to read from PDS file.\n");
        return ApiIOError;
    }
    if (PIXELINK_DATA_STREAM_MAGIC_NUMBER != magicNumber) {
        printf("A valid pds file was not passed.\n");
        return ApiInvalidParameterError;
    }
    
    // number of frames in the pds file
    U32 numberOfFramesInFile;
    // next field is the number of frames in the pds file.
    numItemsRead = fread(
        (void*)&numberOfFramesInFile,
        sizeof(numberOfFramesInFile),
        1,
        hFile);
    if (numItemsRead != 1) {
        printf("IOERROR: Failed to read from PDS file.\n"); 
        return ApiIOError;
    }
    if (numberOfFramesInFile == 0) {
        printf("PDS file does not contain any frames.\n");
        return ApiSuccess;
    }

    // position to the first FRAME_DESC (where we currently are)
    const long int OFFSET_FIRST_FRAME_DESC = ftell(hFile);
    if (-1L == OFFSET_FIRST_FRAME_DESC) {
        printf("IOERROR: Could not record position of first frame description in "
            "PDS binary.\n");
        return ApiIOError;
    }

    // size of the FRAME_DESC (by reading the first 4 byte field from FRAME_DESC)
    U32 sizeOfFrameDesc = 0;
    numItemsRead = fread(
        (void*)&sizeOfFrameDesc,
        sizeof(sizeOfFrameDesc),
        1,
        hFile);
    if (numItemsRead != 1) {
        printf("IOERROR: Failed to read from PDS file.\n");
        return ApiIOError;
    }
    if (sizeOfFrameDesc == 0) {
        printf("ERROR: FRAME_DESC size found to be zero, PDS file may be corrupted.\n");
        return ApiIOError;
    }

    // buffer for frame description data
    vector<U8> frameDescBuffer(sizeOfFrameDesc);
    // convenience pointer to the beginning of the frame description buffer
    FRAME_DESC* pFrameDesc = (FRAME_DESC*)&frameDescBuffer[0];

    // we now know the size of FRAME_DESC, so well return to the start of the
    // first FRAME_DESC and read in the full description data

    int seekResult = fseek(hFile, OFFSET_FIRST_FRAME_DESC, SEEK_SET);
    if (seekResult != 0) {
        printf("IOERROR: Failed to seek to first frame position in PDS file.\n");
        return ApiIOError;
    }
    numItemsRead = fread((void*)pFrameDesc, frameDescBuffer.size(), 1, hFile);
    if (numItemsRead != 1) {
        printf("IOERROR: Failed to read from PDS file.\n");
        return ApiIOError;
    }

    // get the necessary metadata

    // decimation (e.g. binned pixels for the frames)
    const U32 decimation = (U32)pFrameDesc->Decimation.fValue; 
    // frame width in pixels
    const U32 frameWidth = (U32)pFrameDesc->Roi.fWidth / decimation;
    // frame height in pixels
    const U32 frameHeight = (U32)pFrameDesc->Roi.fHeight / decimation;
    // num pixels
    const U32 numPixels = frameWidth * frameHeight; 
    // frame size as cv::Size obj
    const Size cvFrameSize(frameWidth, frameHeight); 
    // frame rate
    const double frameRate = pFrameDesc->FrameRate.fValue;
    // beginning frame time
    const double initFrameTime = pFrameDesc->dFrameTime;
    // beginning frame number
    const UINT64 initFrameNumber = pFrameDesc->u64FrameNumber;
    
    // pixel data size (in bytes)
    const float pixelDataSize = GetPixelSize((U32)pFrameDesc->PixelFormat.fValue);
    if ((U32)pixelDataSize == 0) {
        printf("ERROR: unexpected value (0) for pixelDataSize recieved.\n");
        return ApiIOError;
    }
    // frame data size (in bytes)
    const U32 frameDataSize = (U32)((float)numPixels * pixelDataSize);
    if (frameDataSize == 0) {
        printf("ERROR: unexpected value (0) for frameDataSize recieved.\n");
        return ApiIOError;
    }

    // frame buffer for holding binary image data
    vector<U8> frameBuffer(frameDataSize);
    // convenience pointer to the beginnning of the frame data buffer
    void* pFrame = (void*)&frameBuffer[0];

    // let's pre-compute the size of the loaded image data
    numItemsRead = fread(
        pFrame,
        frameBuffer.size(),
        1,
        hFile);
    if (numItemsRead != 1) {
        printf("IOERROR: Failed to read from PDS file.\n");
        return ApiIOError;
    }
    // conversion image format
    const U32 IMAGE_CONVERSION_FORMAT = IMAGE_FORMAT_RAW_BGR24;
    // size of the raw image data (in bytes) after converson from binary
    U32 rawImageDataSize = 0; 
    returnCode = PxLFormatImage(
        pFrame,
        pFrameDesc,
        IMAGE_CONVERSION_FORMAT,
        NULL,
        &rawImageDataSize);
    if (!API_SUCCESS(returnCode)) {
        printf("ERROR: encountered error when pre-compting for raw image"
            " data conversion.\n");
        return returnCode;
    }
    // buffer to hold the raw image
    vector<U8> rawImageBuffer(rawImageDataSize);

    // now that we have everything we need, lets seek back to the first frame

    seekResult = fseek(hFile, OFFSET_FIRST_FRAME_DESC, SEEK_SET); 
    if (seekResult != 0) {
        printf("IOERROR: Failed to seek to first frame position in PDS file.\n"); 
        return ApiIOError; 
    }

    // a cv matrix for holding the image data, initialize with zeroes
    Mat rawImageMatRGB = Mat(cvFrameSize, CV_8UC3, &rawImageBuffer[0]);

    // a cv matrix for putting annotations
    Mat textImageMatRGB = Mat(Size(frameWidth, 45 + 8), CV_8UC3, Scalar(0, 0, 0));

    // cv matrix for the combined image with annotation
    Mat fullMat;
    Size fullMatSize(frameWidth, frameHeight + 45 + 8); // FIXME - hardcoded


    // deleting any existing files
    char timeFilePath[MAX_PATH];
    sprintf_s(
        timeFilePath,
        sizeof(timeFilePath),
        "%s-TIME.csv",
        pOutputFilePath);

    bool isErr;
    isErr = filesystem::remove(pOutputFilePath);
    if (isErr) {
        printf("WARNING: Deleted existing converted video file; will overwrite.\n");
    }
    isErr = filesystem::remove(timeFilePath);
    if (isErr) {
        printf("WARNING: Deleted existing video timing file; will overwrite.\n");
    }

    printf("Opening a video writer for avi creation.\n");

    // video codec
    const int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    // video params
    const vector<int> writerParams{
        VIDEOWRITER_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY};
    // VideoWriter for creating avi output file
    VideoWriter outputVideo(  
        pOutputFilePath, 
        codec, 
        frameRate,  
        fullMatSize,
        writerParams);
    bool isSupported = outputVideo.set(VIDEOWRITER_PROP_QUALITY, 100);
    if (!isSupported) {
        printf("WARNING: Could not set property quality for video writer.\n");
    }
    waitKey(100);

    if (!outputVideo.isOpened()) {
        printf("ERROR: Failed to open video file.\n");
        return -1;
    }     

    printf("VideoWriter backend utilized is: \"%s\".\n", 
        outputVideo.getBackendName().c_str());
    printf("VideoWriter quality is set to: %f\n",
        outputVideo.get(VIDEOWRITER_PROP_QUALITY));
    printf("VideoWriter acceleration is set to: %f\n",
        outputVideo.get(VIDEOWRITER_PROP_HW_ACCELERATION));

    // setup file for writing frame time
    fstream timeFile;   
    timeFile.open(
        timeFilePath, 
        ios_base::out | ios_base::trunc);
    char lineBuffer[256];

    if (timeFile.is_open()) {
        sprintf_s(
            lineBuffer,
            sizeof(lineBuffer),
            "%16s, %16s, %32s",
            "frame-index", "frame-number", "time-nanosecond");
        timeFile << lineBuffer << "\n";
    } else {
        printf("ERROR: Failed to open time file.");
        timeFile.close();
        return -1;
    }

    // create a window for previewing the output
    printf("Creating a preview window.\n");

    // TODO - have name reflect the currently converted video

    namedWindow("Preview Window", WINDOW_AUTOSIZE);
    waitKey(100);

    double lastFrameTime, liveFps; 
    UINT64 frameTimeNanosec, correctedFrameNumber;
    lastFrameTime = (1.0 / frameRate) * -1.0;

    // main conversion loop
    for (U32 i = 0; i < numberOfFramesInFile; i++) {
        // read in the current frame description and frame data
        numItemsRead = fread(
            (void*)pFrameDesc,
            frameDescBuffer.size(),
            1,
            hFile);
        if (numItemsRead != 1) {
            printf("IOERROR: Failed to read from PDS file.\n");
            timeFile.close();
            return ApiIOError;
        }
        numItemsRead = fread(
            pFrame,
            frameBuffer.size(), 
            1, 
            hFile); 
        if (numItemsRead != 1) {
            printf("IOERROR: Failed to read from PDS file.\n");
            timeFile.close();
            return ApiIOError;
        }

        // this--by updating the data in rawImageBuffer--will 
        // also update the data which rawImageMatRGB references
        returnCode = PxLFormatImage(
            pFrame, 
            pFrameDesc, 
            IMAGE_CONVERSION_FORMAT, 
            (void*)&rawImageBuffer[0], 
            &rawImageDataSize); 
        if (!API_SUCCESS(returnCode)) {
            printf("ERROR: encountered error when converting binary image data.\n");
            timeFile.close();
            return returnCode;
        }

        // computed properties
        correctedFrameNumber = pFrameDesc->u64FrameNumber - initFrameNumber;
        liveFps = 1 / (pFrameDesc->dFrameTime - initFrameTime - lastFrameTime);
        lastFrameTime = pFrameDesc->dFrameTime - initFrameTime;
        frameTimeNanosec = static_cast<UINT64>(
            pFrameDesc->dFrameTime * static_cast<double>(1E9));

        // append line to text file
        memset(lineBuffer, 0, sizeof(lineBuffer));
        sprintf_s(
            lineBuffer,
            sizeof(lineBuffer),
            "%16d, %16llu, %32llu",
            i, pFrameDesc->u64FrameNumber, frameTimeNanosec);
        timeFile << lineBuffer << "\n";

        // add info text to bottom of frame
                
        // buffer for annotation strings
        char frameStr[128];
        sprintf_s(frameStr,  
            sizeof(frameStr),
            "INDEX: %07d | FRAME: %07llu | DROP: %07llu",
            i,
            correctedFrameNumber,
            correctedFrameNumber - static_cast<U64>(i));
        textImageMatRGB.setTo(Scalar(0, 0, 0)); // clear writing
        putText(
            textImageMatRGB,
            frameStr,
            Point(6, 22),
            FONT_HERSHEY_DUPLEX,
            0.6,
            Scalar(255, 255, 255));
        memset(frameStr, 0, sizeof(frameStr));

        // TODO - add experimental info to label on video

        sprintf_s(frameStr,  
            sizeof(frameStr),
            "TIME: %04d:%06.3f m:s | FPS: %7.3f",
            static_cast<int>(lastFrameTime / 60.0),
            lastFrameTime - (floor(lastFrameTime / 60.0) * 60.0), 
            liveFps);
        putText(
            textImageMatRGB,
            frameStr,
            Point(6, 45),
            FONT_HERSHEY_DUPLEX,
            0.6,
            Scalar(255, 255, 255));
        vconcat(rawImageMatRGB, textImageMatRGB, fullMat);

        if (i % 100 == 0) {
            printf("Processing image %6d of %6d ...\n", i, numberOfFramesInFile);
            // printf("Frame time (ns): %32llu\n", frameTimeNanosec);                              
        }

        if (i % 10 == 0) {
            imshow("Preview Window", fullMat);
            waitKey(1);
        }

        outputVideo.write(fullMat); 
    }

    timeFile << "\n";
    timeFile.close();
    
    printf("Conversion complete!\n");

    return returnCode;
}


//
// Given the pixel format, return the size of a individual pixel (in bytes)
// Taken from the getsnapshot demo app (getsnapshot.c).
//
// Returns 0 on failure.
//
static float
GetPixelSize(const U32 pixelFormat)
{
    float retVal = static_cast<float>(0);

    switch (pixelFormat)
    {
    case PIXEL_FORMAT_MONO8:
    case PIXEL_FORMAT_BAYER8_BGGR:
    case PIXEL_FORMAT_BAYER8_GBRG:
    case PIXEL_FORMAT_BAYER8_GRBG:
    case PIXEL_FORMAT_BAYER8_RGGB:
        return 1.0f;
    case PIXEL_FORMAT_MONO16:
    case PIXEL_FORMAT_BAYER16_BGGR:
    case PIXEL_FORMAT_BAYER16_GBRG:
    case PIXEL_FORMAT_BAYER16_GRBG:
    case PIXEL_FORMAT_BAYER16_RGGB:
    case PIXEL_FORMAT_YUV422:
        return 2.0f;
    case PIXEL_FORMAT_RGB24_DIB:
    case PIXEL_FORMAT_RGB24_NON_DIB:
    case PIXEL_FORMAT_BGR24_NON_DIB:
        return 3.0f;
    case PIXEL_FORMAT_RGBA:
    case PIXEL_FORMAT_BGRA:
    case PIXEL_FORMAT_ARGB:
    case PIXEL_FORMAT_ABGR:
        return 4.0f;
    case PIXEL_FORMAT_RGB48:
        return 6.0f;
    case PIXEL_FORMAT_MONO12_PACKED:
    case PIXEL_FORMAT_BAYER12_BGGR_PACKED:
    case PIXEL_FORMAT_BAYER12_GBRG_PACKED:
    case PIXEL_FORMAT_BAYER12_GRBG_PACKED:
    case PIXEL_FORMAT_BAYER12_RGGB_PACKED:
    case PIXEL_FORMAT_MONO12_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST:
        return 1.5f;
    case PIXEL_FORMAT_MONO10_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST:
    case PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST:
        return 1.25f;
    case PIXEL_FORMAT_STOKES4_12:
    case PIXEL_FORMAT_POLAR4_12:
    case PIXEL_FORMAT_POLAR_RAW4_12:
    case PIXEL_FORMAT_HSV4_12:
        return 6.0f;

    }

    return retVal;
}