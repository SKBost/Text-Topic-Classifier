public class Pair<T1, T2> {
    T1 first;
    T2 second;

    public Pair(T1 newFirst, T2 newSecond) {
        first = newFirst;
        second = newSecond;
    }

    public void setFirst(T1 newFirst) {
        first = newFirst;
    }

    public T1 getFirst() {
        return first;
    }

    public void setSecond(T2 newSecond) {
        second = newSecond;
    }

    public T2 getSecond() {
        return second;
    }
}