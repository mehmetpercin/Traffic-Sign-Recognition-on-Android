package com.mp.camera2app;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;

public class ImageClassifier {
    private final String TAG = this.getClass().getSimpleName();

    private static final String MODEL_PATH = "model.tflite";
    private static final String LABEL_PATH = "class_names.txt";

    private static final int PIXEL_SIZE = 3;

    static final int IMG_SIZE_X = 32;
    static final int IMG_SIZE_Y = 32;

    private static final int IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255f;

    private int[] intValues = new int[IMG_SIZE_X * IMG_SIZE_Y];

    private Interpreter tflite;

    private List<String> labelList;

    private ByteBuffer imgData;

    private float[][] labelProbArray;
    private float[][] filterLabelProbArray ;
    private static final int FILTER_STAGES = 5;
    private static final float FILTER_FACTOR = 0.4f;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    1,
                    (o1, o2) -> (o1.getValue()).compareTo(o2.getValue()));

    ImageClassifier(Activity activity) throws IOException {
        tflite = new Interpreter(loadModelFile(activity));
        labelList = loadLabelList(activity);
        imgData = ByteBuffer.allocateDirect(4 * IMG_SIZE_X * IMG_SIZE_Y * PIXEL_SIZE);//4=float size
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][labelList.size()];
        filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    }

    String classifyFrame(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG,"Model could not load");
            return "Uninitialized Classifier.";
        }
        convertBitmapToByteBuffer(bitmap);
        tflite.run(imgData, labelProbArray);

        applyFilter();

        String textToShow = printTopKLabels();
        return textToShow;
    }

    void applyFilter(){
        int numLabels =  labelList.size();

        for(int i=0; i<numLabels; i++){
            filterLabelProbArray[0][i] += FILTER_FACTOR*(labelProbArray[0][i] -
                    filterLabelProbArray[0][i]);
        }

        for (int i=1; i<FILTER_STAGES; i++){
            for(int j=0; j<numLabels; j++){
                filterLabelProbArray[i][j] += FILTER_FACTOR*(
                        filterLabelProbArray[i-1][j] - filterLabelProbArray[i][j]);
            }
        }

        for(int i=0; i<numLabels; i++) {
            labelProbArray[0][i] = filterLabelProbArray[FILTER_STAGES - 1][i];
        }
    }

    public void close() {
        tflite.close();
        tflite = null;
    }

    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < IMG_SIZE_X; i++) {
            for (int j = 0; j < IMG_SIZE_Y; j++) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
    }

    private String printTopKLabels() {
        for (int i = 0; i < labelList.size(); i++) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            if (sortedLabels.size() > 1) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; i++) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = label.getKey();
        }
        return textToShow;
    }
}
