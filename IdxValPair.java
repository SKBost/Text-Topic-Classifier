public class IdxValPair implements Comparable<IdxValPair> {
    private int idx;
    private double val;

    public IdxValPair(int newIdx, double newVal) {
        idx = newIdx;
        val = newVal;
    }

    public int getIdx() {
        return idx;
    }

    public double getVal() {
        return val;
    }

    @Override
    public int compareTo(IdxValPair other) {
        double comparison = this.getVal() - other.getVal();
        if(comparison > 0) {
            return 1;
        } else if(comparison == 0) {
            return 0;
        } else {
            return -1;
        }
    }
}