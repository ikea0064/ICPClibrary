package ICPClibrary;

import java.util.*;  //Scanner, ArrayListなど
import java.awt.geom.*; //Point2D, Line2Dなど
import java.io.*;      //BufferedReaderなど
import java.util.regex.*;//Matcherなど　正規表現を多用するときは使わない
import static java.lang.Math.*; //static importの仕方。僕はあまり好きじゃない

public class Template {
	
	//static連発をしないようにする
	//EOF読み込みなら while(sc.hasNext()){
	//dataset読み込みなら while(dataset-- > 0){
	private void doit(){
		Scanner sc = new Scanner(System.in);
		while(true){
			int n = sc.nextInt();
			int m = sc.nextInt();
			if((n|m) == 0) break;
		}
	}
	
	//デバッグ用。ほとんどなんでもつっこめる
	private void debug(Object... o) {
		System.out.println("debug = " + Arrays.deepToString(o));
	}
	
	//入力読み込み。Scannerより高速
	class InStream{
		BufferedReader in;StringTokenizer st;
		public InStream() {
			this.in = new BufferedReader(new InputStreamReader(System.in));
			this.st = null;
		}
		String next() {
			while (st==null || !st.hasMoreTokens()) {
				try {
					st = new StringTokenizer(in.readLine());
				} catch (Exception e) {}
			}
			return st.nextToken(); 
		}
		boolean hasNext(){
			try{
				st = new StringTokenizer(in.readLine());
				return true;
			}catch(Exception e){return false;}
		}
		int nextInt() {	return Integer.parseInt(next());}
		long nextLong() {return Long.parseLong(next());}
		double nextDouble() {return Double.parseDouble(next());}
	}
	
	//Topcoder用入力読み取り
	class TCInStream{
		private Scanner sc;
		public TCInStream() {
			this.sc = new Scanner(System.in);
		}
		
		private boolean hasNext(){
			return sc.hasNext();
		}
		
		private String [] nextArray(){
			StringBuilder sb = new StringBuilder();
			while(true){
				String s = sc.next().trim();
				sb.append(s);
				if(s.indexOf("}") >= 0) break;
			}
			String s = sb.toString().replaceAll("[{}\" ]", "");
			String [] t = s.split(",");
			return t;
		}
		
		private String next() {
			String s = sc.next().replaceAll("\"", "");
			return s;
		}
		
		private int[] nextIntArray() {
			String [] t = nextArray();
			int [] res = new int[t.length];
			for(int i = 0; i < t.length; i++){
				res[i] = Integer.parseInt(t[i]);
			}
			return res;
		}
		
		private int nextInt() {
			return Integer.parseInt(next());
		}
		
		private long nextLong() {
			return Long.parseLong(next());
		}
	}
	
	public static void main(String[] args) {
		new Template().doit();
	}
}
