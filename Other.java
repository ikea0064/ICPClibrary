package ICPClibrary;
import java.io.*;
import java.util.*;
import java.util.regex.*;

/*
クラスライブラリ
・サイコロ・サイコロの等価比較・構文解析
数論
・エラトステネスのふるい・最大公約数・最小公倍数・バイナリサーチ・素因数分解・べき剰余
・行列累乗・行列積・ガウスの消去法・ライツアウト（ガウスの消去法）
データ構造
・union-find
グラフ, フロー
・MST(prim, kruskal)・ワーシャルフロイド・ベルマンフォード・負の閉路検出・ダイクストラの経路復元
・オイラー路判定・２部マッチング・最小費用流・最大流・強連結成分分解
文字列
・ローリングハッシュ・最長回文(N^2)
ゲーム理論
・nim
日付と時間
・経過日数・ゼイラーの公式・秒を変換・日数の配列・うるう年判定
そのた
・部分文字列・切り上げ切り捨て・４回転・正規表現・八方向座標
日本語
倍数判定（）・重複組合せ・
 */

public class Other {
	
	int INF = 1 << 24;
	double EPS = 10.e-08;
	
	//======クラスライブラリ========
	
	//サイコロ	//dice is, 2 = south, 3 = east, 1 = top, 6 = bottom, 4 = west, 5 = north;
	class Dice{
		int [] dice;
		public Dice(int[] dice) {
			this.dice = dice;
		}
		private void slide() {
			swap(2,4,5,3);
		}
		private void turn(char c) {
			switch(c){
			case 'n': swap(1,5,6,2); break;
			case 's': swap(1,2,6,5); break;
			case 'w': swap(1,4,6,3); break;
			case 'e': swap(1,3,6,4); break;
			}
		}
		private void swap(int i, int j, int k, int l) {
			int temp = dice[l];
			dice[l] = dice[k];
			dice[k] = dice[j];
			dice[j] = dice[i];
			dice[i] = temp;
		}
	}
	
	//サイコロの等価判定
	private boolean isSame(Dice aa, Dice bb) {
		String [] order = {"", "n", "s", "w", "e", "nn"};
		String [] revorder = {"", "s", "n", "e", "w", "nn"};
		Dice a = new Dice(aa.dice);
		Dice b = new Dice(bb.dice);
		for(int i = 0; i < order.length ; i++){
			
			for(int j = 0; j < order[i].length(); j++){
				char c = order[i].charAt(j);
				b.turn(c);
			}
			
			for(int j = 0; j < 4; j++){
				b.slide();
				if(comp(a,b)){
					return true;
				}
			}
			
			for(int j = 0; j < revorder[i].length(); j++){
				char c = revorder[i].charAt(j);
				b.turn(c);
			}
		}
		return false;
	}
	
	private boolean comp(Dice a, Dice b) {
		for(int i = 1; i <= 6; i++){
			if(a.dice[i] != b.dice[i]) return false;
		}
		return true;
	}
	
	//StringBuilderサイコロ	//dice is, 1 = south, 2 = east, 0 = top, 5 = bottom, 3 = west, 4 = north;
		class StringDice{
			StringBuilder dice;
			public StringDice(String dice) {
				this.dice = new StringBuilder(dice);
			}
			private void slide() {
				swap(1,3,4,2);
			}
			private void turn(char c) {
				switch(c){
				case 'n': swap(0,4,5,1); break;
				case 's': swap(0,1,5,4); break;
				case 'w': swap(0,3,5,2); break;
				case 'e': swap(0,2,5,3); break;
				}
			}
			private void swap(int i, int j, int k, int l) {
				char temp = dice.charAt(l);
				dice.setCharAt(l, dice.charAt(k));
				dice.setCharAt(k, dice.charAt(j));
				dice.setCharAt(j, dice.charAt(i));
				dice.setCharAt(i, temp);
			}
		}
	
	//構文解析（電卓）
	//strの末端には終端文字列を入れる(#など)
	class Parse{
		String str;
		int pos;
		public Parse(String str){
			this.str = str;	pos = 0;
		}
		
		private int exp() {
			int res = term();
			while(true){
				char op = str.charAt(pos);
				if((op == '+') || (op == '-')){
					int old = res;
					pos++;
					res = term();
					switch(op){
					case '+': 
						res = old + res; break;
					case '-':
						res = old - res; break;
					}
				}
				else break;
			}
			return res;
		}

		private int term() {
			int res = fact();
			while(true){
				char op = str.charAt(pos);
				if((op == '*') || (op == '/')){
					int old = res;
					pos++;
					res = fact();
					switch(op){
					case '*':
						res = old * res; break;
					case '/':
						res = old / res; break;
					}
				}
				else break;
			}
			return res;
		}

		private int fact() {
			if(Character.isDigit(str.charAt(pos))){
				int t = str.charAt(pos) - '0';
				pos++;
				while(Character.isDigit(str.charAt(pos))){
					t = t * 10 + (str.charAt(pos) - '0');
					pos++;
				}
				return t;
			}
			else if(str.charAt(pos) == '('){
				pos++;
				int res = exp();
				pos++;
				return res;
			}
			return 0;
		}
	}
	
	//エラトステネスのふるい
	private void eratos(){
		int MAX = 1000000;
		boolean [] isprime = new boolean[MAX + 1];
		Arrays.fill(isprime, true);
		isprime[0] = false;
		isprime[1] = false;
		for(int i = 0; i * i <= MAX; i++){
			if(isprime[i]){
				for(int j = i * 2; j <= MAX; j+= i){
					isprime[j] = false;
				}
			}
		}
	}
	
	//最大公約数
	//O(log max(a,b))
	private int gcd(int a, int b){
		if(b == 0)  return a;
		else        return gcd(b, a%b);
	}
	
	//最小公倍数
	private int lcm(int a, int b){
		return a / gcd(a, b) * b;
	}
	
	//バイナリサーチ
	//添字番号を返す。見つからないなら-1
	//同じ数字が複数あるならば、どの数字の添字番号が選ばれるかわからない
	private int binarySearch(int [] a, int patt){
		int mid, left = 0, right = a.length-1;
		while(left <= right){
			mid = (left + right) / 2;
			if(a[mid] == patt){
				return mid;
			}
			if(a[mid] < patt){
				left = mid + 1;
			}
			else{
				right = mid - 1;
			}
		}
		return -1;
	}
	
	//バイナリサーチ
	//添字番号を返す。見つからないなら-1
	//複数同じ数字がある場合は添字番号が小さいものを返す
	private int bLowerSearch(int a[], int patt){
		int mid, left = 0, right = a.length - 1;
		while(left < right){
			mid = (right + left) / 2;
			if(a[mid] < patt){
				left = mid + 1;
			}
			else{
				right = mid;
			}
		}
		if(a[left] == patt){
			return left;
		}
		else{
			return -1;
		}
	}

	//素因数分解
	private int[] primeDecomposition(int n) {
		int [] res = new int[n+1];
		for(int i=2; i * i <= n; i++){
			while(n % i == 0){
				++res[i];
				n /= i;
			}
		}
		if(n != 1) res[n] = 1;
		return res;
	}
	
	//a^eのmod値を返す。logn
	private int modpow(int a, int e, int mod) {
		long result = 1;
		while(e > 0){
			if((e&1) == 1){
				result = (result * a) % mod;
			}
			e >>= 1;
			a = (a * a) % mod;
		}
		return (int)result;
	}
	
	//行列累乗。aのn乗の行列を返す。O(log n)
	private int[][] matrixPow(int[][] a, int n) {
		int [][] b = new int[a.length][a[0].length];
		for(int i = 0 ; i < a.length; i++){
			b[i][i] = 1;
		}
		while( n > 0){
			if((n & 1) != 0){
				b = mul(b , a);
			}
			a = mul(a,a);
			n >>= 1;
		}
		return b;
	}

	//行列演算。a * bの行列を返す
	private int[][] mul(int[][] a, int[][] b) {
		int MOD = 10000;
		int [][] c = new int[a.length][b[0].length];
		for(int i = 0 ; i < a.length; i++){
			for(int k = 0 ; k < b.length; k++){
				for(int j = 0 ; j < b[0].length; j++){
					c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % MOD;
				}
			}
		}
		return c;
	}
	
	//ガウスの消去法
	//解がないか、一意でない時はnullを返す
	private double [] gauss_jordan(double  [][] A, double [] b){
		int n = A.length;
		double [][] B = new double[n][n + 1];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				B[i][j] = A[i][j];
			}
		}
		
		for(int i = 0; i < n; i++){
			B[i][n] = b[i];
		}
		
		for(int i = 0; i < n; i++){
			int pivot = i;
			for(int j = i; j < n; j++){
				if(Math.abs(B[j][i]) > Math.abs(B[pivot][i])) pivot = j;
			}
			for(int j = 0; j <= n; j++){
				double temp = B[i][j];
				B[i][j] = B[pivot][j];
				B[pivot][j] = temp;
			}
			
			if(Math.abs(B[i][i]) < EPS) return null;
			
			for(int j = i + 1; j <= n; j++) B[i][j] /= B[i][i];
			for(int j = 0; j < n; j++){
				if(i != j){
					for(int k = i + 1; k <= n; k++){
						B[j][k] -= B[j][i] * B[i][k];
					}
				}
			}
		}
		double [] res = new double[n];
		for(int i = 0; i < n; i++){
			res[i] = B[i][n];
		}
		return res;
	}
	
	//ライツアウトの解があるかどうかを返す
	//a[i][j] = j番目のボタンを押した時にi番目のボタンに影響があるなら1が格納されている
	//a[i][n]には、現在の状態が格納されている
	//実際にはmod2の連立方程式を解くだけなので、xorの計算でよい
	//最後の方で[i][i] == 0で[i][n] == 0ならば解が不定、0以外なら解なし
	private boolean lightsOut(int[][] a) {
		int n = a.length;
		for(int i = 0; i < n; i++){
			int pivot = i;
			for(int j = i; j < n; j++){
				if(Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
			}
			
			if(Math.abs(a[pivot][i]) == 0) continue;
			
			for(int j = 0; j <= n; j++){
				int temp = a[i][j];
				a[i][j] = a[pivot][j];
				a[pivot][j] = temp;
			}
			
			for(int j = 0; j < n; j++){
				if(i != j && a[j][i] != 0){
					for(int k = i; k <= n; k++){
						a[j][k] ^= a[i][k];
					}
				}
			}
		}
		for(int i = 0; i < n; i++){
			if(a[i][i] == 0 && a[i][n] != 0){
				return false;
			}
		}
		return true;
	}
	
	//======データ構造=====
	
	//union-find木。グループ管理ができる。挿入、検索にlogNかかる（たぶん）
	//find();ルートナンバーを返す。unite();グループを結合する。num;グループ数を返す
	//uniteする時は、ランクが低いルートが新しいルートになる
	class UnionFind{
		int [] par, rank;
		int num;
		public UnionFind(int n) {
			par = new int[n];
			rank = new int[n];
			for(int i = 0 ; i < n; i++){
				par[i] = i;
				//rank[i] = 0;
			}
			num = n;
		}
		int find(int x){
			if(par[x] == x)return x;
			else return par[x] = find(par[x]);
		}
		void unite(int x, int y){
			x = find(x);
			y = find(y);
			if(x == y) return ;
			if(rank[x] < rank[y]) par[x] = y;
			else{
				par[y] = x;
				if(rank[x] == rank[y]) rank[x]++;
			}
			num--;
		}
		boolean same(int x, int y){
			return find(x) == find(y);
		}
	}
	
	//======グラフ=====
	
	//辺を状態を格納するクラス
	class Edge implements Comparable<Edge>{
		int from,to, cost;
		public Edge(int from, int to, int cost) {
			this.from = from; this.to = to; this.cost = cost;
		}
		public int compareTo(Edge o) {
			if(this.cost < o.cost)return -1;
			if(this.cost > o.cost)return 1;
			return 0;
		}
	}
	//ダイクストラの状態クラス
	class State implements Comparable<State>{
		int now, cost;
		public State(int now, int cost) {
			this.now = now;
			this.cost = cost;
		}
		public int compareTo(State o) {
			return this.cost - o.cost;
		}
	}
	
	//MST //O ( nodesize^2 )
	private int mst(int[][] pass) {
		int n = pass.length;
		int [] mincost = new int[n];
		boolean [] isused = new boolean[n];
		Arrays.fill(mincost, INF);
		mincost[0] = 0;
		int res = 0;
		while(true){
			int v = -1;
			for(int i = 0; i < n; i++){
				if(! isused[i] &&( v == -1 || mincost[i] < mincost[v])){
					v = i;
				}
			}
			if(v == -1) break;
			isused[v] = true;
			res += mincost[v];
			for(int i = 0; i < n; i++){
				mincost[i] = Math.min(mincost[i], pass[v][i]);
			}
		}
		return res;
	}
	
	//クラスカル法 O(m log m) m-> edgesize
	private int kruskal(ArrayList<Edge> edgelist, int nodesize) {
		int m = edgelist.size();
		Collections.sort(edgelist);
		UnionFind uf = new UnionFind(nodesize);
		int res = 0;
		for(int i = 0 ; i < m; i++){
			Edge e = edgelist.get(i);
			if(! uf.same(e.from, e.to)){
				uf.unite(e.from, e.to);
				res += e.cost;
			}
		}
		return res;
	}
	
	//ワーシャルフロイド
	//pass[i][i]には0を格納しておく
	//pass[i][i]がマイナスになっていれば負の閉路がある
	private int[][] warshallFloyd(int [][] pass){
		int n = pass.length;
		for(int j = 0; j < n; j++){
			for(int i = 0; i < n; i++){
				for(int k = 0; k < n; k++){
					pass[i][k] = Math.min(pass[i][k], pass[i][j] + pass[j][k]);
				}
			}
		}
		return pass;
	}
	
	//ベルマンフォード
	private int [] bellman(ArrayList<Edge> data, int n, int s){
		int m = data.size();
		int [] d = new int[n];
		Arrays.fill(d, INF);
		d[s] = 0;
		while(true){
			boolean update = false;
			for(int i = 0; i < m; i++){
				Edge e = data.get(i);
				if(d[e.from] != INF && d[e.to] > d[e.from] + e.cost){
					d[e.to] = d[e.from] + e.cost;
					update = true;
				}
			}
			if(! update) break;
		}
		return d;
	}
	
	//負の閉路検出
	//最短経路は高々n-1回の更新しかあり得ないので、n回目の更新があった時は負の閉路となる
	private boolean find_negative_loop(ArrayList<Edge> data, int n, int s){
		int [] d = new int[n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < data.size(); j++){
				Edge e = data.get(j);
				if(d[e.to] > d[e.from] + e.cost){
					d[e.to]= d[e.from] + e.cost;
					
					if(i == n - 1) return true;
				}
			}
		}
		return false;
	}
	
	//隣接行列を用いた経路復元のダイクストラ。経路を返す。たぶんO(E log V)
	private String dijkstra(int s, int g, ArrayList<ArrayList<Edge>> pass) {
		PriorityQueue<State> open = new PriorityQueue<State>();
		open.add(new State(s, 0));
		int n = pass.size();
		double [] close = new double[n];
		Arrays.fill(close, INF);
		close[s] = 0.0;
		int [] route = new int[n];
		route[s] = -1;
		boolean flg = false;
		while(! open.isEmpty()){
			State now = open.poll();
			if(now.now == g){
				flg = true;
				break;
			}
			for(int i = 0 ; i < pass.get(now.now).size(); i++){
				Edge nowe = pass.get(now.now).get(i);
				int nextcost = now.cost + nowe.cost;
				if(close[nowe.to] <= nextcost) continue;
				close[nowe.to] = nextcost;
				open.add(new State(nowe.to, nextcost));
				route[nowe.to] = nowe.from;
			}
		}
		if(! flg){
			return "NA";
		}
		StringBuilder ans = new StringBuilder();
		int prev = g;
		LinkedList<Integer> stack = new LinkedList<Integer>();
		while(prev >= 0){
			stack.add(prev + 1);
			prev = route[prev];
		}
		while(! stack.isEmpty()){
			ans.append(" " + stack.removeLast());
		}
		return ans.substring(1);
	}
	
	//無向グラフオイラー路判定, IntegerはEdgeが格納されている。無向グラフ専用
	//無向グラフの場合,すべてのノードの次数（ノードに接続するエッジ数）が偶数（この時はオイラー閉路）または
	//始点と終点のノードの次数が奇数で、残りのノードの次数が偶数
	//有向グラフの場合,すべてのノードで入ってくるエッジ数と出ていくエッジ数が同じ（この時はオイラー閉路）または
	//始点は入ってくるエッジ数よりも出ていくエッジ数が1多い,終点は入ってくるエッジ数が出ていくエッジ数よりも1多い
	private boolean isEuler(ArrayList<ArrayList<Integer>> pass,int start, int goal) {
		for(int i = 0; i < pass.size(); i++){
			if(i == start || i == goal) continue;
			if(pass.get(i).size() % 2 != 0){
				return false;
			}
		}
		if(pass.get(start).size() % 2 == pass.get(goal).size() % 2){
			if(pass.get(start).size() % 2 == 1){
				return true; //オイラー路
			}
			else{
				return true; //オイラー閉路
			}
		}
		return false;
	}
	
	//最小費用流
	class MCF{
		ArrayList<ArrayList<Edge>> G;
		class Edge {
			int to, cap, cost, rev;
			public Edge(int to, int cap, int cost, int rev) {
				this.to = to;this.cap = cap;this.cost = cost; this.rev = rev;
			}
		}
		
		MCF(int v){
			G = new ArrayList<ArrayList<Edge>>();
			for(int i = 0; i < v; i++){
				G.add(new ArrayList<Edge>());
			}
		}
		private void addEdge(int from, int to, int cap, int cost){
			G.get(from).add(new Edge(to, cap, cost, G.get(to).size()));
			G.get(to).add(new Edge(from, 0, -cost, G.get(from).size() - 1));
		}
		private int minCostFlow(int s, int t, int f) {
			int V = G.size();
			int [] dist = new int[V], prevv = new int[V], preve = new int[V];
			int res = 0;
			while(f > 0){
				Arrays.fill(dist, INF);
				dist[s] = 0;
				boolean update = true;
				while(update) {
					update = false;
					for(int v = 0; v < V; v++){
						if(dist[v] == INF) continue;
						for(int i = 0 ; i < G.get(v).size(); i++){
							Edge  e = G.get(v).get(i);
							if(e.cap > 0 && dist[e.to]> dist[v] + e.cost ){
								dist[e.to] = dist[v] + e.cost;
								prevv[e.to] = v;
								preve[e.to] = i; 
								update = true;
							}
						}
					}
				}
				if(dist[t] == INF) return -1;
				
				int d = f;
				for(int v = t; v != s; v = prevv[v]){
					d = Math.min(d, G.get(prevv[v]).get(preve[v]).cap);
				}
				f -= d;
				res += d * dist[t];
				for(int v = t; v!= s; v = prevv[v]){
					Edge e =G.get(prevv[v]).get(preve[v]);
					e.cap -= d;
					G.get(v).get(e.rev).cap += d;
				}
			}
			return res;
		}
	}
	//２部マッチング
	class BM{
		//隣接リスト。from,toのみ。重みは存在しない
		//m,nとするとき、nはm+n_iに入れる
		ArrayList<ArrayList<Integer>> pass;
		boolean [] used;
		int [] match;
		BM(ArrayList<ArrayList<Integer>> pass){
			this.pass =pass;

			//２部マッチング開始
			int ans = 0;
			Arrays.fill(match, -1);
			for(int i=0; i < pass.size(); i++){
				if(match[i] < 0){
					Arrays.fill(used, false);
					if(dfs(i)){
						ans++;
					}
				}
			}
			System.out.println(ans);
		}

		private boolean dfs(int v){
			used[v] = true;
			for(int i =0; i < pass.get(v).size(); i++){
				int u = pass.get(v).get(i);
				int w = match[u];
				if(w < 0 || !used[w] && dfs(w)){
					match[v] = u;
					match[u] = v;
					return true;
				}
			}
			return false;
		}
	}
	//最大流 事前にcapを入れておく
	class MF{
		int INF = 1 << 24;
		int n , s, t;
		int [][] cap, flow;
		int [] levels;
		boolean [] finished;
		
		MF(int [][] cap, int n, int s, int t){
			this.cap = cap;
			this.n = n; this.s = s; this.t = t;
			flow = new int[n][n];
			finished = new boolean[n];
			levels = new int[n];
		}
		
		private int maxflow() {
			for(int i = 0 ; i < n; i++) Arrays.fill(flow[i], 0);
			int total = 0;
			for(boolean cont = true; cont; ){
				cont = false;
				levelize();
				Arrays.fill(finished, false);
				for(int f; (f = augment(s,INF)) > 0; cont = true)
					total += f;
			}
			return total;
		}
		private void levelize() {
			Arrays.fill(levels, -1);
			LinkedList<Integer> q = new LinkedList<Integer>();
			levels[s] = 0;
			q.add(s);
			while(! q.isEmpty()){
				int here = q.removeFirst();
				for(int there = 0; there < n; there++){
					if(levels[there] < 0 && residue(here, there) > 0){
						levels[there] = levels[here] + 1;
						q.add(there);
					}
				}
			}
		}
		private int augment(int here, int cur) {
			if(here == t || cur == 0)
				return cur;
			if(finished[here])
				return 0;
			finished[here] = true;
			for(int there = 0; there < n; there++){
				if(levels[there] > levels[here]){
					int f = augment(there, Math.min(cur, residue(here, there)));
					if(f > 0){
						flow[here][there] += f;
						flow[there][here] -= f;
						finished[here] = false;
						return f;
					}
				}
			}
			return 0;
		}
		
		private int residue(int i, int j) {
			return cap[i][j] - flow[i][j];
		}
	}
	//強連結成分分解
	//事前準備でグラフを生成して、関数を行う。連結後の頂点数を返す。所属はcmpを参照する
	class SCC{
		int n;
		ArrayList<ArrayList<Integer>> g, rg;
		ArrayList<Integer> vs;
		boolean [] used;
		int [] cmp;
		
		SCC(int n){
			this.n = n;
			g = new ArrayList<ArrayList<Integer>>();
			rg = new ArrayList<ArrayList<Integer>>();
			for(int i = 0 ; i < n; i++){
				g.add(new ArrayList<Integer>());
				rg.add(new ArrayList<Integer>());
			}
			vs = new ArrayList<Integer>();
			used = new boolean[n];
			cmp = new int[n];
		}
		private void addEdge(int from, int to){
			g.get(from).add(to);
			rg.get(to).add(from);
		}
		private void dfs(int v){
			used[v] = true;
			for(int i = 0; i < g.get(v).size(); i++){
				if(! used[g.get(v).get(i)]) dfs(g.get(v).get(i));
			}
			vs.add(v);
		}
		private void rdfs(int v , int k){
			used[v] = true;
			cmp[v] = k;
			for(int i = 0; i < rg.get(v).size(); i++){
				if(! used[rg.get(v).get(i)]) rdfs(rg.get(v).get(i), k);
			}
		}
		private int scc(){
			Arrays.fill(used, false);
			for(int v = 0; v < n; v++){
				if(! used[v] ) dfs(v);
			}
			Arrays.fill(used, false);
			int k = 0 ;
			for(int i = vs.size() - 1; i >= 0; i--){
				if(!used[vs.get(i)]) rdfs(vs.get(i), k++);
			}
			System.out.println(Arrays.toString(cmp));
			return k;
		}
	}
	
	//=========文字列==========
	//[a,b]間のローリングハッシュ値をもとめる(a以上b以下)
	//hsは累積のハッシュ値が入っている
	//qは基数のn乗的なものが入っている
	//累積和の差分を引くときに、qでケタを合わせないとダメなんだと思う（よくわからない）
	private void rollingHash(String str, int a, int b){
		long H = 1000000007L;
		int n = str.length();
		char [] s = str.toCharArray();
		long[] hs = new long[n+1];
		hs[0] = 0;
		for(int i = 0;i < n;i++)
			hs[i+1] = (hs[i] * H + (s[i]-'a'+1));
		long[] q = new long[n+1];
		q[0] = 1;
		for(int i = 1;i <= n;i++){
			q[i] = q[i-1] * H;
		}
		long hh = hs[b+1]-hs[a]*q[b+1-a];
	}
	
	//最長回文 O(N^2)
	private int longestPalindrome(String s){
		int len = s.length();
		int res = 0;
		for(int pos = 0 ; pos < len; pos++){
			int i = 0;
			for(i = 0; pos + i < len && pos - i - 1 >= 0; i++){
				char begin = s.charAt(pos - i - 1) ;
				char end = s.charAt(i + pos);
				if(begin != end){
					break;
				}
			}
			i--;
			res = Math.max(res, (i+1)* 2);
			for(i = 0; pos + i < len && pos - i >= 0; i++){
				char begin = s.charAt(pos - i);
				char end = s.charAt(i + pos);
				if(begin != end){
					break;
				}
			}
			i--;
			res = Math.max(res, i * 2 + 1);
		}
		return res;
	}
	
	//=========ゲーム理論==========
	
	//nim
	//石の山がn個あって、それぞれai個の石を含んでいます。2人は交互に空でない山を一つ選び
	//そこから1つ以上の石を取ります。どちらが勝つのかを求める問題
	//XORを取ればいいらしいけど、何故なのかはよく分かっていない
	private String calcNim(ArrayList<Integer> pile){
		int nim = 0;
		for(int value: pile){
			nim = nim ^ value;
		}
		if(nim == 0){
			return "LOSE";
		}
		else{
			return "WIN";
		}
	}
	
	//=========日付==========
	//1年1月1日からの経過日数を求めるプログラム
	int getdays(int y, int m, int d) {
		if (m <= 2) {
			y--;
			m += 12;
		}
		int dy = 365 * (y - 1); // 経過年数×365日
		int c = y / 100;
		int dl = (y >> 2) - c + (c >> 2); // うるう年分
		int dm = (m * 979 - 1033) >> 5; // 1月1日から m 月1日までの日数
		return dy + dl + dm + d - 1;
	}
	
	//ゼイラーの公式
	//西暦を入力として曜日を返す。0が日曜日
	int zeller(int y, int m, int d) {
		if (m <= 2) {
			y--;
			m += 12;
		}
		return (y + y/4 - y/100 + y/400 + (13 * m + 8) / 5 + d) % 7;
	}
	
	//秒の変換
	private int [] totime(int time){
		int h = time / 3600;
		int m = (time - 3600 * h) / 60;
		int s = time - 3600 * h - 60 * m;
		//int time = h * 3600 + m * 60 + s // reverse
		return new int[]{h,m,s};
	}
	
	//日数の配列
	private int[] dayOfMonth(){
		int [] day = {-1,31,28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
		return day;
	}
	
	//うるう年かどうかを判定する
	private boolean isleapYear(int i) {
		if(i % 4 != 0){
			return false;
		}
		else if(i % 400 == 0){
			return true;
		}
		else if(i % 100 == 0){
			return false;
		}
		else{
			return true;
		}
	}
	
	//=========その他==========
	//部分文字列
	//substring(beginidx, endidx)   // ("ikeya").substring(1, 3) => "ke"
	
	//切り上げ => Math.ceil, 切り捨て => Math.floor
	
	//4回転。正方行列のみ
	private int[][][] rotate(int [][] data){
		int n = data.length;
		int [][][] mlist = new int[4][n][n];
		for(int i = 0 ; i < n; i++){
			for(int j = 0 ; j < n; j++){
				mlist[0][i][j] = data[i][j];
			}
		}
		for(int  i = 1 ; i < 4; i++){
			for(int y = 0 ; y < n; y++){
				for(int x = 0; x < n; x++){
					mlist[i][x][n- y - 1] = mlist[i-1][y][x];
				}
			}
		}
		return mlist;
	}
	
	//正規表現の前方参照
	private void howToRegex(){
		String str = "iikeyaa";
		String regex = "(.*)ikeya(.*)";   //部分一致の時のやり方。必要に応じて?をつける
		Pattern p = Pattern.compile(regex);

		Matcher m = p.matcher(str);
		if (m.find()){
			String matchstr = m.group();
			System.out.println(matchstr +  " is matched");
			System.out.println(m.group(0));   // iikeyaa
			System.out.println("group1:" + m.group(1));  // i
			System.out.println("group2:" + m.group(2));  // a
		}
	}
	
	//八方向座標
	//上、右、下、左、右上、右下、左下、左上
	private void eightDirection() {
		int [] vx = {0,1,0,-1,1,1,-1,-1};
		int [] vy = {1,0,-1,0,1,-1,-1,1};
	}
	
	//=========日本語==========
	
	//倍数判定
	/*
	２；一の位が2の倍数
	３；各位の和が3の倍数
	４；下2桁が4の倍数
	５；一の位が5の倍数
	６；2かつ3の倍数
	７；３桁毎に交互に足したり引いたりしてできた数が７の倍数
	   ３桁の数 ａｂｃ で、ａｂ-２ｃ が７の倍数
	８；下３桁が８の倍数
	　　一の位を２で割り十の位に足して２で割った数を百の位に足した数が偶数
	９；各位の数の和が９の倍数
	１０；一の位が０
	１１；各位の数を交互に足したり引いたりしてできた数が１１の倍数
	１２；３かつ４の倍数
	１３；７の倍数の判定と同じ
	１６；下４桁を２で割った数が８の倍数（下４桁を４で割った数が４の倍数）
	１７；十位以上の数から一位の数の５倍を引いた数が１７の倍数
	　　２桁毎に下位から２のべきを掛けて交互に足したり引いたりしてできた数が１７の倍数
	１８；２かつ９の倍数
	１９；各位の数に上位から２のべきを掛けて足した数が１９の倍数
	２０；４かつ５の倍数
	２１；３かつ７の倍数
	２２；２かつ１１の倍数
	２３；十位以上の数と一位の数の７倍の和が２３の倍数
	２４；３かつ８の倍数
	３７；３桁毎に区分けした数を足した数が３７の倍数
	９９９；３桁毎に区分けした数を足した数が９９９の倍数
	*/
	
	//重複組み合わせ
	//重複組み合わせとは，ある数を複数使用して良い組み合わせのこと
	//nHr = n+r-1Cr
}
