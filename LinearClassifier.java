import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifier {

    private final int numFeatures;
    private ArrayList<DataPoint> trPts; // training data points with feature vectors and labels
    private ArrayList<DataPoint> allTrPts; // set of all points, 1 vs ... vs 6 (with labels of the real classes)
    private ArrayList<DataPoint> oneVsTwoTrPts; // class 1 vs 2 (with labels -1 and 1)

    private ArrayList<DataPoint> oneVsAllTrPts; // class 1 vs others
    private ArrayList<DataPoint> twoVsAllTrPts; // class 2 vs others
    private ArrayList<DataPoint> threeVsAllTrPts; // class 3 vs others
    private ArrayList<DataPoint> fourVsAllTrPts; // etc...
    private ArrayList<DataPoint> fiveVsAllTrPts;
    private ArrayList<DataPoint> sixVsAllTrPts;

    private ArrayList<DataPoint> testPts;
    private ArrayList<DataPoint> allTestPts;
    private ArrayList<DataPoint> oneVsTwoTestPts;

    private ArrayList<DataPoint> oneVsAllTestPts;
    private ArrayList<DataPoint> twoVsAllTestPts; // class 2 vs others
    private ArrayList<DataPoint> threeVsAllTestPts; // class 3 vs others
    private ArrayList<DataPoint> fourVsAllTestPts; // etc...
    private ArrayList<DataPoint> fiveVsAllTestPts;
    private ArrayList<DataPoint> sixVsAllTestPts;

    private double[] pw3; // perceptron's weight vector after 3 iterations
    private double[] lrw50; // logistic regression's weight vector after 50 iterations of gradient descent 

    private static double placeholder = 420.69;

    public LinearClassifier(int newNumFeatures, String trainFile, String testFile) {
        numFeatures = newNumFeatures;
        allTrPts = readDataFrom(trainFile);
        allTestPts = readDataFrom(testFile);
        trPts = allTrPts;
        testPts = allTestPts;

        initTrPts();
        initTestPts();

        pw3 = null;
        lrw50 = null;
    }

    private void initTrPts() {
        ArrayList<DataPoint> oneVsTwoPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> oneVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> twoVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> threeVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> fourVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> fiveVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> sixVsAllPts = new ArrayList<DataPoint>();

        for(int i = 0; i < allTrPts.size(); i++) {
            DataPoint tp = allTrPts.get(i);

            // constructing oneVsTwoTrPts
            if(tp.getLabel() == 1 || tp.getLabel() == 2) {
                double newLabel;
                if(tp.getLabel() == 1) {
                    newLabel = -1;
                } else {
                    newLabel = 1;
                }
                DataPoint oneVsTwoPt = new DataPoint(tp.getFeatures(), newLabel);
                oneVsTwoPts.add(oneVsTwoPt);
            }

            // constructing all other xVsAllTrPts (-1 means it is a member of the named type)
            if(tp.getLabel() == 1) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 2) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 3) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 4) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 5) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 6) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
            } else {
                print("error: label was not a number 1-6");
            }

        }

        oneVsTwoTrPts = oneVsTwoPts;
        oneVsAllTrPts = oneVsAllPts;
        twoVsAllTrPts = twoVsAllPts;
        threeVsAllTrPts = threeVsAllPts;
        fourVsAllTrPts = fourVsAllPts;
        fiveVsAllTrPts = fiveVsAllPts;
        sixVsAllTrPts = sixVsAllPts;
    }

    private void initTestPts() {
        ArrayList<DataPoint> oneVsTwoPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> oneVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> twoVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> threeVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> fourVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> fiveVsAllPts = new ArrayList<DataPoint>();
        ArrayList<DataPoint> sixVsAllPts = new ArrayList<DataPoint>();

        for(int i = 0; i < allTestPts.size(); i++) {
            DataPoint tp = allTestPts.get(i);

            // constructing oneVsTwoTrPts
            if(tp.getLabel() == 1 || tp.getLabel() == 2) {
                double newLabel;
                if(tp.getLabel() == 1) {
                    newLabel = -1;
                } else {
                    newLabel = 1;
                }
                DataPoint oneVsTwoPt = new DataPoint(tp.getFeatures(), newLabel);
                oneVsTwoPts.add(oneVsTwoPt);
            }

            // constructing all other xVsAllTrPts (-1 means it is a member of the named type)
            if(tp.getLabel() == 1) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 2) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 3) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 4) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 5) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
            } else if(tp.getLabel() == 6) {
                oneVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                twoVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                threeVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fourVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                fiveVsAllPts.add(new DataPoint(tp.getFeatures(), 1));
                sixVsAllPts.add(new DataPoint(tp.getFeatures(), -1));
            } else {
                print("error: label was not a number 1-6");
            }

        }

        oneVsTwoTestPts = oneVsTwoPts;
        oneVsAllTestPts = oneVsAllPts;
        twoVsAllTestPts = twoVsAllPts;
        threeVsAllTestPts = threeVsAllPts;
        fourVsAllTestPts = fourVsAllPts;
        fiveVsAllTestPts = fiveVsAllPts;
        sixVsAllTestPts = sixVsAllPts;
    }

    public void useClasses1And2() {
        trPts = oneVsTwoTrPts;
        testPts = oneVsTwoTestPts;
    }

    public void useAllClasses() {
        trPts = allTrPts;
        testPts = allTestPts;
    }

    public void useNumVsAll(int num) {
        if(num == 1) {
            trPts = oneVsAllTrPts;
            testPts = oneVsAllTestPts;
        } else if(num == 2) {
            trPts = twoVsAllTrPts;
            testPts = twoVsAllTestPts;
        } else if(num == 3) {
            trPts = threeVsAllTrPts;
            testPts = threeVsAllTestPts;
        } else if(num == 4) {
            trPts = fourVsAllTrPts;
            testPts = fourVsAllTestPts;
        } else if(num == 5) {
            trPts = fiveVsAllTrPts;
            testPts = fiveVsAllTestPts;
        } else if(num == 6) {
            trPts = sixVsAllTrPts;
            testPts = sixVsAllTestPts;
        } else {
            print("error: invalid input for useNumVsAll; choose a num between 1 and 6");
        }
    }

    public int[][] getConfusionMatrix(/*double[] classifier, Array<DataPoints> testData*/) {
        int[][] confusionMatrix = new int[7][6]; 
        /*
        a confusion matrix is a 6×6 matrix, where each row is labelled 1, . . . , 6
        and each column is labelled 1, . . . , 6. The entry of the matrix at row i and column j is Cij/Nj where
        Cij is the number of test examples that have label j but are classified as label i by the classifier, and
        Nj is the number of test examples that have label j. Since the one-vs-all classifier can also predict
        Don’t Know, the confusion matrix will now be an 7 × 6 matrix – that is, it will have an extra row
        corresponding to the Don’t Know predictions.
        */
        
        ArrayList<Map<DataPoint, Double>> mapList = new ArrayList<Map<DataPoint, Double>>();

        // get classifiers for each class for oneVsAll
        double[][] classifiers = new double[6][numFeatures];
        for(int i = 1; i <= 6; i++) {
            useNumVsAll(i);
            classifiers[i-1] = teachPerceptron(new double[numFeatures]);
        }

        for(int i = 0; i < 6; i++) {
            mapList.add(new HashMap<DataPoint, Double>());
        }
        //= new HashMap<String, Integer>(); 

        // for each possible class
        for(int classNum = 1; classNum <= 6; classNum++) {
            useNumVsAll(classNum);
            Map<DataPoint, Double> map = mapList.get(classNum-1);
            // for each point classified under that way
            for(int i = 0; i < testPts.size(); i++) {
                DataPoint tp = testPts.get(i);
                double prediction = getPerceptronClassification(classifiers[classNum-1], tp);
                // add to map of point to label what prediction there is
                map.put(allTestPts.get(i), prediction);//todo: see if this works to get the right label if I put allTestPts?
            }
        }

        useAllClasses();

        // for each point
        for(int i = 0; i < testPts.size(); i++) {
            DataPoint tp = testPts.get(i);
            // have a count of how many predicted -1
            int count = 0;
            // have a variable representing one class that predicted -1
            int matchClass = 7; // default value for if there are no matches
            // for each class
            for(int classNum = 1; classNum <= 6; classNum++) {
                // get its predicted label from the map
                Map<DataPoint, Double> map = mapList.get(classNum - 1);
                //print("map.size() is " + map.size());

                //for (Map.Entry<DataPoint, Double> me : map.entrySet()) { 
                //    System.out.print(me.getKey().getLabel() + ":"); 
                //    System.out.println(me.getValue()); 
                //} 

                if(map.get(tp) != null) {
                    //print("map.get(tp) is NOT null");
                    double prediction = map.get(tp);
                    // if it is -1, increment the count
                    if(prediction == -1.0) {
                        count++;
                        // if the count is > 1, return a don't know TODO make sure classnum carries through
                        if(count > 1) {
                            matchClass = 7;
                            break;
                        } else {
                            // otherwise, set the class number and continue
                            matchClass = classNum;
                        }
                    }
                } else {
                    print("ERROR: map.get(tp) is null; need to check that tp is used correctly");
                }
            }


            //print("for i = " + i + ", matchClass is " + matchClass + "; actual label is " + tp.getLabel());

            confusionMatrix[matchClass-1][(int)(tp.getLabel())-1]++;

        }
        
/*
        for(int i = 0; i < testPts.size(); i++) {
            DataPoint tp = testPts.get(i);
            double label = tp.getLabel();
            getPerceptronClassification(classifier, tp);

            // this array will hold whether or not the class = index + 1 was predicted as the label
            boolean[] predictedSingleClass = new boolean[6];
            for(int j = 1; j <= 6; j++) {
                useNumVsAll(j);
                double prediction = getPerceptronClassification(classifier, tp);
                if(prediction == -1.0) {
                    predictedSingleClass[j-1] = true; // j indices are one above their usual
                }
            }

            for(int j = 0; j < predictedSingleClass.length; j++) {

            }
            //if(prediction != label) {
            //    confusionMatrix[i][j]++;
            //}
        }
        */

        for(int i = 0; i < confusionMatrix.length; i++) {
            print(Arrays.toString(confusionMatrix[i]));
        }
        return confusionMatrix;
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
            double trError = getPerceptronTrainingError(wT);
            double testError = getPerceptronError(wT, testPts);
            if(i + 1 == 2 || i + 1 == 3 || i + 1 == 4 || i ==0) {
                print("after " + (i + 1) + " runs of teaching perceptron, we get the following training error: " 
                    + trError + " and the following test error: " + testError);
            }

            if(i + 1 == 3) { // classifier after 3 passes
                pw3 = wT;
            }

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

    

    // w is the classifier (todo: change name) VITODO determine which form is correct!
    // perceptron gives correct classification
    private boolean perceptronGivesCorrectClassification(double[] w, DataPoint testDP) {
        
        double yt = testDP.getLabel();
        double[] xt  = testDP.getFeatures();
        double classVal = yt * getDotProduct(w, xt); // classification value
        return classVal > 0;
        

        /*
        double label = testDP.getLabel();
        int classification = getPerceptronClassification(w, testDP);
        //print("label is " + label + " and classification is " + classification + "; do they match? " + (label == ((double) classification)));
        return label == ((double) classification);
        */
        
    }

    private int getPerceptronClassification(double[] w, DataPoint testDP) {
        double yt = testDP.getLabel();
        double[] xt  = testDP.getFeatures();
        double dotProd = getDotProduct(w, xt); // classification value
        if(dotProd > 0) {
            return 1;
        } else {
            return -1;
        }
        //return !(classVal <= 0);
    }

    public double getPerceptronTrainingError(double[] classifier) {
        return getPerceptronError(classifier, trPts);
    }

    public double[] getPW3() {
        return pw3;
    }

    public double[] getLRW50() {
        return lrw50;
    }

    public Pair<IdxValPair[], IdxValPair[]> getExtremeCoords(double[] vector, int numIdxs) {
        IdxValPair[] highPairs = new IdxValPair[numIdxs];
        IdxValPair[] lowPairs = new IdxValPair[numIdxs];
        
        IdxValPair[] vecPairs = new IdxValPair[vector.length];

        for(int i = 0; i < vector.length; i++) {
            IdxValPair pair = new IdxValPair(i, vector[i]);
            vecPairs[i] = pair;
        }

        Arrays.sort(vecPairs);

        //print("now printing vecPairs");
        //for(int i = 0; i < vecPairs.length; i++) {
        //    print("vecPairs[" + i + "] = (" + vecPairs[i].getIdx() + ", " + vecPairs[i].getVal() + ")");
        //}

        for(int i = 0; i < numIdxs; i++) {
            lowPairs[i] = vecPairs[i];
            highPairs[i] = vecPairs[vecPairs.length - 1 - i ];
        }

        print("now printing lowPairs");
        for(int i = 0; i < lowPairs.length; i++) {
            print("lowPairs[" + i + "] = (" + lowPairs[i].getIdx() + ", " + lowPairs[i].getVal() + ")");
        }

        print("now printing highPairs");
        for(int i = 0; i < highPairs.length; i++) {
            print("highPairs[" + i + "] = (" + highPairs[i].getIdx() + ", " + highPairs[i].getVal() + ")");
        }

        Pair<IdxValPair[], IdxValPair[]> lowNHigh = new Pair<IdxValPair[], IdxValPair[]>(lowPairs, highPairs);
        



        /*
        for(int i = 0; i < sortedVec.length; i++) {
            print("sortedVec[" + i + "] is " + sortedVec[i]);
        }
        */
        
        return lowNHigh;
    }

    // end of perceptron methods



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
        double[][] w = new double[numIterations + 1][numFeatures]; // todo: do we actually use numPoints + 1?
        w[0] = new double[numFeatures];
        //print("each w is size " + w[0].length);
        //print("there are " + w.length + " ws");
        //print("before gradient descent, w0 is " + norm(w[0]));
        for(int t = 0; t < numIterations; t++) { // VITDO: CHECK IF THIS IS EVEN SUPPOSED TO RUN THIS MANY TIMES?

            double[] deltaLW = getDeltaLW(w[t]);
            //if(deltaLW.equals(new double[w[t].length])) {
            //    print("getDeltaLW did a bad job");
            //}
            double[] update = multiplyScalarAndVector(rate, deltaLW);
            //if(update.equals(new double[w[t].length])) {
            //    print("getDeltaLW did a bad job");
            //} else {
                //print("norm of update is " + norm(update));
                //printDoubleArray(update);
            //}
            w[t+1] = subtractVectors(w[t], update);
            //if(w[t+1].equals(new double[w[t].length])) {
                //print("subtractVectors did a bad job");
            //} else {
                //print("norm of w[" + (t+1) + "] is " + norm(w[t+1]));
                //printDoubleArray(w[t+1]);
            //}
            if(norm(deltaLW) <= stopThreshold) {
                //print("we return early because norm(deltaLW) is " + norm(deltaLW));
                return w[t + 1];
            } 
            //print("norm(deltaLW) is " + norm(deltaLW));

            // t + 1 is the number of iterations that have been completed
            if(t+1 == 2 || t+1 == 10 || t+1 == 50 || t+1 == 100) { 
                double trError = getLogisticRegressionTrainingError(w[t+1]);
                double testError = getLogisticRegressionError(w[t+1], testPts);
                print("after " + (t+1) + " iterations, the training error is " 
                    + trError + " and the test error is " + testError);

                if(t+1 == 50) {
                    lrw50 = w[t+1];
                }
            }


        }

        //print("trPts.size() is " + trPts.size());
        //print("we do not return early");
        return w[numIterations]; // final form of w after running on last data point
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
            double[] vecI = multiplyScalarAndVector(getScalarToMult(yi, xi, w), vecNumer);
            vecSum = sumVectors(vecSum, vecI);
        }

        /*if(vecSum.equals(new double[w.length])) {
            print("the problem is in deltaLW");
        } else {
            print("vecSum is: ");
            printDoubleArray(vecSum);
        }*/

        return vecSum;
    }

    // this sigmoid functuon gives P(y | x)
    private double sigmoid(double y, double[] x, double[] w) {
        //print("in sigmoid function; y is " + y);
        double dotProd = getDotProduct(w, x);
        //print("dotProd is " + dotProd);
        double exponent = -1 * y * dotProd;
        //print("exponent is " + exponent);
        double sigScal = 1/(1 + Math.exp(exponent)); // sigmoid scalar
        //print("sigScal is " + sigScal);
        return sigScal;
    }

    // vitodo: check on negative sign in loss function vs fancy l
    private double getScalarToMult(double y, double[] x, double[] w) {
        //print("in sigmoid function; y is " + y);
        double dotProd = getDotProduct(w, x);
        //print("dotProd is " + dotProd);
        double exponent = y * dotProd;
        //print("exponent is " + exponent);
        double sigScal = 1/(1 + Math.exp(exponent)); // sigmoid scalar
        //print("sigScal is " + sigScal);
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
        //print("classVal is " + classVal);

        if(classVal == 0.5) { //TODO: find a better way of handling this
            return yt == 1;
        } else {
            return classVal > 0.5;
        }
    }

    public double getLogisticRegressionTrainingError(double[] classifier) {
        return getLogisticRegressionError(classifier, trPts);
    }

    private double getLogisticRegressionError(double[] classifier, ArrayList<DataPoint> testData) {
        //print("final classifier's norm is " + norm(classifier));
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