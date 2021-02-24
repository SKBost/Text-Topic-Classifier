import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifier {

    private final int numFeatures;
    private ArrayList<DataPoint> trPts; // training data points with feature vectors and labels
    private ArrayList<DataPoint> allTrPts;
    private ArrayList<DataPoint> binaryTrPts;
    private ArrayList<DataPoint> testPts;

    private static double placeholder = 420.69;

    public LinearClassifier(int newNumFeatures, String trainFile, String testFile) {
        numFeatures = newNumFeatures;
        allTrPts = readDataFrom(trainFile);
        testPts = readDataFrom("pa3test.txt");
        trPts = allTrPts;

        ArrayList<DataPoint> binaryPts = new ArrayList<DataPoint>();
        for(int i = 0; i < allTrPts.size(); i++) {
            DataPoint tp = allTrPts.get(i);
            if(tp.getLabel() == 1 || tp.getLabel() == 2) {
                double newLabel;
                if(tp.getLabel() == 1) {
                    newLabel = -1;
                } else {
                    newLabel = 1;
                }
                DataPoint binaryPt = new DataPoint(tp.getFeatures(), newLabel);
                binaryPts.add(binaryPt);
            }
        }
        binaryTrPts = binaryPts;
    }

    public void useClasses1And2() {
        trPts = binaryTrPts;
    }

    public void useAllClasses() {
        trPts = allTrPts;
    }

    // tested
    private static double[] multiplyScalarAndVector(double scalar, double[] vector) {
        //print("inside multiplyScalarAndVector");
        double[] newVec = new double[vector.length];
        for(int i = 0; i < vector.length; i++) {
            newVec[i] = vector[i] * scalar;
        }
        /*
        print("multiplying scalar " + scalar  + "and the following vector:");
        print("");
        printDoubleArray(vector);
        print("multiplied vector is ");
        print("");
        printDoubleArray(newVec);
        */
        return newVec;
    }

    // tested
    private double[] sumVectors(double[] first, double[] second) {
        //print("inside sumVectors");

        double[] newVec = new double[first.length];
        
        if(first.length != second.length) {
            print("ERROR in sumVectors: first.length != second.length; returning placeholder zero vector");
        } else {
            for(int i = 0; i < first.length; i++) {
                newVec[i] = first[i] + second[i];
            }
        }

        return newVec;
    }

    // tested
    private double getDotProduct(double[] wt, double[] xt) {
        //print("inside getDotProduct");
        // check if inputs are valid
        if(wt.length != xt.length) {
            print("ERROR in getDotProduct: wt.length != xt.length; returning placeholder " + placeholder);
            return placeholder;
        }

        // get inner/dot product
        double multDimSum = 0;
        for(int i = 0; i < wt.length; i++) {
            multDimSum += wt[i] * xt[i];
            //print("multDimSum (now " + multDimSum + ") = previous multDimSum + wt[" + i +"] (" + wt[i] + ") * xt[" + i + "](" + xt[i] + ")");
        }
        return multDimSum;
    }

  // perceptron

    // runs a single pass of perceptron given a starting value
    public double[] teachPerceptron(double[] w0) { // note: we start at w0 instead of w1 for 0 indexing
    //print("inside teachPerceptron");
    double[][] w = new double[trPts.size() + 1][numFeatures]; // todo: do we actually use numPoints + 1?
    w[0] = w0;
    //print("each w is size " + w[0].length);
    //print("there are " + w.length + " ws");

    for(int t = 0; t < trPts.size(); t++) {
        double yt = trPts.get(t).getLabel();
        double[] xt  = trPts.get(t).getFeatures();
        double classVal = yt * getDotProduct(w[t], xt); // classification value

        if(classVal <= 0) {
            //print("perceptron got it wrong on training point t = " + t + ", where classVal is " 
                //+ classVal + "; updating w[t+1]");
            w[t+1] = sumVectors(w[t], multiplyScalarAndVector(yt, xt));
        } else {
            //print("perceptron got it right on training point t = " + t + ", where classVal is " 
                //+ classVal + "; not updating w[t+1]");
            w[t+1] = w[t];
        }
    }

    //print("trPts.size() is " + trPts.size());
    return w[trPts.size()]; // final form of w after running on last data point
}

    public void runPerceptronPasses(int numPasses) {
        print("inside runPerceptronPasses");
        double[] w0 = new double[numFeatures]; // starter values of 0 for perceptron to build off
        for(int i = 0; i < numPasses; i++) {
            double[] wT = teachPerceptron(w0); // the final value/output
            double error = getPerceptronTrainingError(wT);
            print("after run " + i + " of teaching perceptron, we get the following training error: " + error);
            error = getPerceptronError(wT, testPts);
            print("after run " + i + " of teaching perceptron, we get the following test error: " + error);
            //printDoubleArray(w0);
            w0 = wT; // the final value/output of each pass is the starter for the next pass
        }
    }

    public double getPerceptronError(double[] classifier, ArrayList<DataPoint> testData) {
        int timesCorrect = 0;
        int timesIncorrect = 0;

        // for each line of data with a feature vector and label
        for(int i = 0; i < testData.size(); i++) {

            DataPoint testDP = testData.get(i);
            double label = testDP.getLabel();

            // classify based on training data with our perceptron
            if(perceptronGivesCorrectClassification(classifier, testDP)) { // if classification is correct
                timesCorrect++; // increment how many times we got it right
            } else {
                timesIncorrect++;
                //print("for line " + i + " of the test data, we " + 
                 //     " incorrectly label it as " + classification + 
                //     " instead of the correct label, " + label);
                    
            }
            
        }

        double error = ((double) timesIncorrect) / testData.size();
        //print("error  = " + error);

        //print("--------------------------------------------");

        return error;
    }

    

    // w is the classifier (todo: change name)
    // perceptron gives correct classification
    private boolean perceptronGivesCorrectClassification(double[] w, DataPoint testDP) {
        double yt = testDP.getLabel();
        double[] xt  = testDP.getFeatures();
        double classVal = yt * getDotProduct(w, xt); // classification value
        return !(classVal <= 0);
    }

    public double getPerceptronTrainingError(double[] classifier) {
        return getPerceptronError(classifier, trPts);
    }



    // logistic regression/gradient descent

    /*public void runIterationsOfGradientDescent(int numIterations) {
        print("inside runGradientDescent");
        double[] w0 = new double[numFeatures]; // starter values of 0 for logistic regression to build off
        for(int i = 0; i < numIterations; i++) {
            double[] wT = runGradientDescent(w0); // the final value/output
            double error = getLogisticRegressionTrainingError(wT);
            print("after run " + i + " of gradient descent, we get the following training error: " + error);
            error = getPerceptronError(wT, testPts);
            print("after run " + i + " of gradient descent, we get the following test error: " + error);
            //printDoubleArray(w0);
            w0 = wT; // the final value/output of each pass is the starter for the next pass
    }*/

    public double[] runGradientDescent(int numIterations) {
            //print("inside runGradientDescent");
        double rate = 0.001; // learning rate/step size
        double stopThreshold = 0.001; // todo check if this is good for this program
        double[][] w = new double[trPts.size() + 1][numFeatures]; // todo: do we actually use numPoints + 1?
        w[0] = new double[numFeatures];
        //print("each w is size " + w[0].length);
        //print("there are " + w.length + " ws");

        for(int t = 0; t < numIterations; t++) { // VITDO: CHECK IF THIS IS EVEN SUPPOSED TO RUN THIS MANY TIMES?

            double[] deltaLW = getDeltaLW(w[t]);
            w[t+1] = subtractVectors(w[t], multiplyScalarAndVector(rate, deltaLW));
            
            if(norm(deltaLW) <= stopThreshold) {
                return w[t + 1];
            } 
        }

        //print("trPts.size() is " + trPts.size());
        return w[trPts.size()]; // final form of w after running on last data point
    }

    // function that gives most of the update for w[t+1]
    private double[] getDeltaLW(double[] w) { 
        // note: may be worth changing variable names to be more descriptive, 
        // but since I'm trying to make them match an algorithm, maybe not

        double[] vecSum = new double[w.length];
        
        for(int i = 0; i < trPts.size(); i++) {
            DataPoint dp = trPts.get(i);
            double[] xi = dp.getFeatures();
            double yi = dp.getLabel();

            
            double[] vecNumer = multiplyScalarAndVector(-yi, xi); // vector numerator
            double[] vecI = multiplyScalarAndVector(sigmoid(yi, xi, w), vecNumer);
            vecSum = sumVectors(vecSum, vecI);
        }

        return vecSum;
    }

    // this sigmoid functuon gives P(y | x)
    private double sigmoid(double y, double[] x, double[] w) {
            double exponent = y * getDotProduct(w, x);
            double sigScal = 1/(1 + Math.exp(exponent)); // sigmoid scalar
            return sigScal;
    }

    // w is the classifier (todo: change name)
    // perceptron gives correct classification
    private boolean logisticRegressionGivesCorrectClassification(double[] w, DataPoint testDP) {
        double yt = testDP.getLabel();
        double[] xt  = testDP.getFeatures();
        // classification value: return true if chances are greater than half that the label is 
        // the correct label (yt) given xt according to the predictor w
        double classVal = sigmoid(yt, xt, w); 
        return classVal > 0.5;
    }

    public double getLogisticRegressionTrainingError(double[] classifier) {
        return getLogisticRegressionError(classifier, trPts);
    }

    private double getLogisticRegressionError(double[] classifier, ArrayList<DataPoint> testData) {
        int timesCorrect = 0;
        int timesIncorrect = 0;

        // for each line of data with a feature vector and label
        for(int i = 0; i < testData.size(); i++) {

            DataPoint testDP = testData.get(i);
            double label = testDP.getLabel();

            // classify based on training data with our perceptron
            if(logisticRegressionGivesCorrectClassification(classifier, testDP)) { // if classification is correct
                timesCorrect++; // increment how many times we got it right
            } else {
                timesIncorrect++;
                //print("for line " + i + " of the test data, we " + 
                 //     " incorrectly label it as " + classification + 
                //     " instead of the correct label, " + label);
                    
            }
            
        }

        double error = ((double) timesIncorrect) / testData.size();
        //print("error  = " + error);

        //print("--------------------------------------------");

        return error;
    }

    // TODO
    private double[] subtractVectors(double[] first, double[] second) {
        //print("inside subtractVectors");

        double[] newVec = new double[first.length];
        
        if(first.length != second.length) {
            print("ERROR in subtractVectors: first.length != second.length; returning placeholder zero vector");
        } else {
            for(int i = 0; i < first.length; i++) {
                newVec[i] = first[i] - second[i];
            }
        }

        return newVec;
    }

    // TODO
    private double norm(double[] vector) {
        double sum = 0;

        for(int i = 0; i < vector.length; i++) {
            sum += vector[i] * vector[i];
        }

        return Math.sqrt(sum);
    }


    public static void printDoubleArray(double[] array) {
        System.out.print("[");
            for(int j = 0; j < array.length; j++) {
                System.out.print(array[j] + ", ");
            }
            System.out.println("]");
    }

    public ArrayList<DataPoint> readDataFrom(String fileName) {
        return readDataFrom(fileName, numFeatures);
    }

    public static ArrayList<DataPoint> readDataFrom(String fileName, int numFeatures) {

        ArrayList<DataPoint> data = new ArrayList<DataPoint>();
        //print("readDataFrom " + fileName);

        try {
            File file = new File(fileName); 
            Scanner sc = new Scanner(file); 
    
            while (sc.hasNextLine()) { // make each line into a data point
                String line = sc.nextLine();
                String[] featuresAndLabel = line.split(" ");

                if(featuresAndLabel.length != numFeatures + 1) {
                    print("error: numFeatures is incorrect");
                    return data;
                } else {
                    //print("numFeatures is correct for this line");
                }

                double[] features = new double[numFeatures];
                for(int i = 0; i < numFeatures; i++) {
                    features[i] = Double.parseDouble(featuresAndLabel[i]);
                }

                double label = Double.parseDouble(featuresAndLabel[numFeatures]);
                //System.out.print(Integer.toString(label));
                DataPoint point = new DataPoint(features, label);
                data.add(point);

            }

        } catch (FileNotFoundException ex) {
            print("Invalid file path given: " + fileName);
        }

        return data;
    }


    private static void print(String text) {
        System.out.println(text);
    }
}

 /*
    public void runMethodTests() {
        //testMSAV(); // good
        //testGDP(); // good
        //testSV(); // good

    }

    private void testMSAV() {
        double scalar = 3.0;
        double[] vector = new double[3];
        vector[0] = 1;
        vector[1] = 2;
        vector[2] = 3;
        double[] newVec = multiplyScalarAndVector(scalar, vector);
        printDoubleArray(newVec);

        // expect: [3, 6, 9]
    }

    private void testGDP() {
        double[] vector1 = new double[3];
        vector1[0] = 1;
        vector1[1] = 2;
        vector1[2] = 3;

        double[] vector2 = new double[3];
        vector2[0] = 4;
        vector2[1] = 5;
        vector2[2] = 6;

        double dp = getDotProduct(vector1, vector2);
        print("" + dp);

        // expect: 32
    }

    private void testSV() {
        double[] vector1 = new double[3];
        vector1[0] = 1;
        vector1[1] = 2;
        vector1[2] = 3;

        double[] vector2 = new double[3];
        vector2[0] = 4;
        vector2[1] = 5;
        vector2[2] = 6;

        double[] newVec = sumVectors(vector1, vector2);
        printDoubleArray(newVec);

        // expect: [5, 7, 9]

    }
    */

      /*
    // this method might be trivial
    private double[] getZeroVector() {
        double[] zeroVec = new double[dimNum];
        return zeroVec;
    }
    */