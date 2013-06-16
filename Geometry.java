package ICPClibrary;
import java.awt.geom.*;
import java.util.*;

public class Geometry {
	/*
<目次>
・おまじない（EPS, comparator）
ベクトル系
・外積・内積・足し算・引き算・かけ算・距離・ノーム・法線ベクトル・ベクトルの角度・CCW
点
・多角形の包含関係・最近点対
線分系
・線と線の交点・線分と点の距離・直線と点の距離・線分と線分の距離・線分と線分の交差判定・射影
角度
・3辺から角度
円
・円と円の交点・円と直線の交点・半径rの弧の長さ
面積
・多角形の面積・ヘロンの公式・ピックの定理・数値積分・解の公式・fx関数
幾何応用
・凸包・凸カット・垂直二等分線・同じかどうか・線分アレンジメント
3次元
・距離・３次元ベクトルの内積・３次元ベクトルの距離・３次元ベクトルの角度
・点・直線・距離・ベクトルの距離・ノルム・足し算・引き算・かけ算・角度・射影・反射・点と直線の距離
そのた
・多角形の辺をmidの値分内側に平行移動させた線分
・円周上の２点から中心点を求める
・y座標からxを計算する関数
	 */
	
	//おまじない。EPS,Point2Dのソート
	final double EPS = 1.0e-08;
	//使うときは、Arrays.sort(data, com);
	Comparator< Point2D > com = new Comparator< Point2D >() {
		public int compare(Point2D o1, Point2D o2) {
			if(o1.getX() < o2.getX()) return -1;
			else if(o1.getX() > o2.getX()) return 1;
			else if(o1.getY() < o2.getY()) return -1;
			else if(o1.getY() > o2.getY()) return 1;
			else return 0;
		}
	};
	
	//外積
	private double cross(Point2D p1, Point2D p2) {
		double res = p1.getX() * p2.getY() - p1.getY() * p2.getX();
		return res;
	}
	
	//内積
	private double dot(Point2D p1, Point2D p2){
		return p1.getX() * p2.getX() + p1.getY() * p2.getY();
	}
	
	//足し算
	private Point2D add(Point2D p1, Point2D p2) {
		double x = p2.getX() + p1.getX();
		double y = p2.getY() + p1.getY();
		return new Point2D.Double(x, y);
	}
	
	//引き算
	private Point2D sub(Point2D p2, Point2D p1) {
		double x = p2.getX() - p1.getX();
		double y = p2.getY() - p1.getY();
		return new Point2D.Double(x, y);
	}
	
	//かけ算。スカラー倍
	private Point2D mul(double value, Point2D p) {
		return new Point2D.Double(p.getX() * value, p.getY() * value);
	}
	
	//ノーム
	private double norm(Point2D p){
		return p.getX() * p.getX() + p.getY() * p.getY();
	}
	
	//距離を求める。3次元のnormはsqrtはいれていない
	private double abs(Point2D p){
		return Math.sqrt(p.getX() * p.getX() + p.getY() * p.getY());
	}
	
	//法線ベクトル
	private Point2D normalVector1(Point2D p) {
		return new Point2D.Double(-p.getY(), p.getX());
	}
	
	//法線ベクトルその２
	private Point2D normalVector2(Point2D p) {
		return new Point2D.Double(p.getY(), -p.getX());
	}
	
	//ベクトルの角度
	private double angle(Point2D p1, Point2D p2){
		double a = dot(p1, p2);
		double b = abs(p1);
		double c = abs(p2);
		double cosTheta = Math.acos(a/b/c);
		return Math.toDegrees(cosTheta);
	}
	
	//CCW
	//直線の左側が-1、右側が１ 線上だと0、線の延長線上はp1に近い方が-1 p2に近い方が1
	private int ccw(Line2D l, Point2D p) {
		return l.relativeCCW(p);
	}
	
	//CCW(Line2D 使用不可)
	//線分p1p2に大してp3のccwを行う。こちらは延長線上は0となる
	private int ccw2(Point2D p1, Point2D p2, Point2D p3){
		Point2D p3p1 = sub(p3, p1);
		Point2D p3p2 = sub(p3, p2);
		double res = cross(p3p1, p3p2);
		if(Math.abs(res) < EPS) return 0;
		else if(res < EPS) return -1;
		else return 1;
	}
	
	//====== 点 =====
	//多角形の包含関係
	//-1 -> out, 0 -> border, 1 -> in
	//verifyしてない.自作サンプルでverify
	private int contains(Point2D[] plist,Point2D p) {
		boolean isin = false;
		int n = plist.length;
		for(int i = 0 ; i < n; i++){
			Point2D a = sub(plist[i], p);
			Point2D b = sub(plist[(i+1) % n], p);
			if(a.getY() > b.getY()){
				Point2D temp = (Point2D) a.clone();
				a = b;
				b = temp;
			}
			if(a.getY() <= 0 && 0 < b.getY()){
				if(cross(a,b) < 0) isin = ! isin;
			}
			if(cross(a,b) == 0 && dot(a, b) <= 0) return 0;
		}
		return isin ? 1 : -1;
	}
	

	//最近点対。平均計算量 nlogn、最悪計算量 n^2
	private Point2D [] closeestPair(Point2D [] p){
		int n = p.length, s = 0, t = 1, m = 2;
		int [] S = new int[n];
		S[1] = 1;
		Arrays.sort(p, com);
		double d = norm(sub(p[s],p[t]));
		for(int i = 2 ; i < n;S[m++] = i++){
			for(int j = 0 ; j < m; j++){
				if(norm(sub(p[S[j]], p[i])) < d){
					d = norm(sub(p[S[j]], p[i]));
					s = S[j];
					t = i;
				}
				if(p[S[j]].getX() < p[i].getX() - d){
					S[j--] = S[--m];
				}
			}
		}
		return new Point2D[]{p[s], p[t]};
	}
	
	//======線分系=====
	
	//線分と線分の交点
	private Point2D intersectPtSS(Line2D l, Line2D m) {
		Point2D lVec = sub(l.getP2(), l.getP1());
		Point2D mVec = sub(m.getP2(), m.getP1());
		Point2D m1l1Vec = sub(m.getP1(), l.getP1());
		double a = cross(m1l1Vec, lVec);
		double b = cross(lVec, mVec);
		if(Math.abs(a) < EPS && Math.abs(b) < EPS){
			//平行な直線同士の場合
			if(l.getP1().distance(m.getP1()) < EPS) return l.getP1();
			if(l.getP1().distance(m.getP2()) < EPS) return l.getP1();
			return l.getP2();
		}
		double t = a / b;
		double resx = m.getX1() + t * mVec.getX();
		double resy = m.getY1() + t * mVec.getY();
		return new Point2D.Double(resx, resy);
	}
	
	//線分と点の距離
	private double distanceSP(Line2D l, Point2D p){
		return l.ptSegDist(p);
	}
	
	//直線と点の距離
	private double distanceLP(Line2D l, Point2D p){
		return l.ptLineDist(p);
	}
	
	//線分と線分の距離
	private double distanceSS(Line2D l, Line2D m){
		double ans = 0.0;
		if(! l.intersectsLine(m)){
			double res1 = l.ptSegDist(m.getP1());
			double res2 = l.ptSegDist(m.getP2());
			double res3 = m.ptSegDist(l.getP1());
			double res4 = m.ptSegDist(l.getP2());
			ans = Math.min(Math.min(res1, res2), Math.min(res3, res4));
		}
		return ans;
	}
	
	//線分と線分の交差判定
	private boolean isIntersectSS(Line2D l, Line2D m) {
		return l.intersectsLine(m);
	}

	//線分から直線にする。値域が広いとき、線分の座標が小数のときは使えない。もしくはベクトルでやるとか
	private Line2D segmentToLine(Line2D l) {
		final int K = 10000;
		Point2D Vec = sub(l.getP2(), l.getP1());
		Point2D p1 = new Point2D.Double(K * Vec.getX() + l.getX1(), K * Vec.getY() + l.getY1());
		Point2D p2 = new Point2D.Double(-K * Vec.getX() + l.getX1(), -K * Vec.getY() + l.getY1());
		return new Line2D.Double(p2, p1);
	}
	
	//射影。直線上にある点pに垂直な点を返す。
	//未verify
	private Point2D projection(Line2D l, Point2D p){
		double t = dot(sub(p, l.getP1()), sub(l.getP1(), l.getP2())) / norm(sub(l.getP1(), l.getP2()));
		return add(l.getP1(), mul(t, sub(l.getP1(), l.getP2())));
	}
	
	//======線分系=====
	
	//3辺から角度A(単位はrad)を求める。（余弦定理）
	private double calcRadA(double a, double b, double c) {
		double costheta = (b * b + c * c - a * a) / (2 * b * c); 
		double res = Math.acos(costheta);
		return res;
	}
	
	//======円=====
	
	//円
	public class Circle{
		Point2D p;
		double r;
		public Circle(Point2D p, double r) {
			this.p = p; this.r = r;
		}
	}
	//円と円の交点
	Point2D [] intersectPtCC(Circle a,Circle b) {
		double dis = a.p.distance(b.p);
		if(dis > a.r + b.r)return null;
		Point2D v = sub(b.p, a.p);
		double rc = (dis * dis + a.r * a.r - b.r * b.r) / (2 * dis);
		double rate = rc / dis;
		v = mul(rate, v);
		Point2D c = add(v, a.p);
		double disC2c = c.distance(b.p);
		double disqc = Math.sqrt(b.r * b.r - disC2c * disC2c);
		Point2D v2 = sub(b.p, c);
		v2 = mul(disqc / disC2c, v2);
		Point2D [] res = new Point2D[2];
		res[0] = add(normalVector1(v2), c);
		res[1] = add(normalVector2(v2), c);
		return res;
	}
	
	//円と直線の交点。線分の場合は線分上にあるかどうかを最後に追加すればよい
	//自作verify
	private Point2D [] intersectPtCL( Circle c, Line2D l ){
		Point2D [] res = new Point2D[2];
		Point2D h = projection(l, c.p);
		double dis = abs(sub(h, c.p));
		if(dis < c.r - EPS){
			//交点が2つのとき
			Point2D dir = sub(l.getP1(), l.getP2());
			Point2D x = mul(1.0 / abs(dir) * Math.sqrt(c.r * c.r - dis * dis), dir);
			res[0] = add(h, x);
			res[1] = sub(h, x);
		}
		else if(dis < c.r + EPS){
			//交点が1つのとき
			res[0] = h;
		}
		return res;
	}
	
	//半径rの弧の長さを求める
	private double calcArc(double r, double rad) {
		return r * rad;
	}
	
	//======面積=====
	//多角形の面積
	//凸でなくても大丈夫。反時計回りに格納されている
	private double area(ArrayList<Point2D> polygon) {
		double res = 0.0;
		int n = polygon.size();
		for(int i = 0; i < n; i++){
			Point2D from = polygon.get(i), to = polygon.get((i+1) % n);
			res += cross(from, to);
		}
		return Math.abs(res) / 2.0;
	}
	
	//ヘロンの公式。３辺から面積を返す。
	private double heron(double a, double b, double c) {
		double z = (a+b+c) / 2;
		double s = z * (z - a) * (z - b) * (z - c);
		return Math.sqrt(s);
	}
	
	//ピックの定理。not verify
	//等間隔に点が存在する平面上にある多角形の面積をもとめる公式
	//多角形の内部にある格子点の個数をi,辺上にある格子点の個数をbとする
	private double pick(int i, int b){
		double s = 0.5 * b + i - 1;
		return s;
	}
	
	//数値積分

	//関数fを使った区間aからbまでのシンプソン公式
	private double simpson(int [] f,double a, double b){
		double nowsum = 0;
		int separatesize = 10000;
		double deltaX = (b - a) / (2 * separatesize);
		nowsum = calcF(f, a) + 4.0 * calcF(f, a + deltaX) + calcF(f, b);
		
		for(int j = 1; j < separatesize; j++){
			nowsum += 2.0 * calcF(f,a + 2 * j * deltaX) + 
					4.0 * calcF(f, a + (2 * j + 1) * deltaX);
		}
		nowsum = deltaX * nowsum / 3.0;
		return nowsum;
	}

	//シンプソン公式に使うときの関数fxの中身。必要に応じて変える
	//この中身は、2次関数の放物線の長さをもとめている
	//sqrt(1 + fx')
	private double calcF(int[] f, double range) {
		double a = 2 * f[0] * range;
		double b = f[1];
		double sq = 1 + (a + b) * (a + b);
		return Math.sqrt(sq);
	}
	
	//蟻本のまる写し。よくわかっていない
	class Integral{
		int m,n;
		int [] x1,x2,y1,z2;
		int INF = 1 << 24;
		
		private void solve() {
			int min1 = minAll(x1);
			int max1 = maxAll(x1);
			int min2 = minAll(x2);
			int max2 = maxAll(x2);
			ArrayList<Integer> xs = new ArrayList<Integer>();
			for(int i = 0 ; i < m; i++){
				xs.add(x1[i]);
			}
			for(int i = 0 ; i < n; i++){
				xs.add(x2[i]);
			}
			Collections.sort(xs);
			
			double res = 0;
			for(int i = 0 ; i + 1 < xs.size(); i++){
				double a = xs.get(i), b = xs.get(i+1), c = (a + b) / 2;
				if(min1 <= c && c <= max1 && min2 <= c && c <= max2){
					double fa = width(x1, y1, m, a) * width(x2, z2, n, a);
					double fb = width(x1, y1, m, b) * width(x2, z2, n, b);
					double fc = width(x1, y1, m, c) * width(x2, z2, n, c);
					res += (b-a) / 6 * (fa + 4 * fc + fb);
				}
			}
			System.out.printf("%.10f\n", res);
		}
		private double width(int[] X, int[] Y, int n, double x) {
			double lb = INF, ub = -INF;
			for(int i = 0 ; i < n; i++){
				double x1 = X[i], y1 = Y[i], x2 = X[ (i + 1) % n], y2 = Y[(i+1) % n];
				if((x1 - x) * (x2 - x) <= 0 && x1 != x2){
					double y = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
					lb = Math.min(lb,  y);
					ub = Math.max(ub, y);
				}
			}
			return Math.max(0.0, ub - lb);
		}
		private int minAll(int[] x) {
			int res = x[0];
			for(int i = 1; i < x.length; i++){
				res = Math.min(res, x[i]);
			}
			return res;
		}
		
		private int maxAll(int[] x) {
			int res = x[0];
			for(int i = 1; i < x.length; i++){
				res = Math.max(res, x[i]);
			}
			return res;
		}
	}

	//2次方程式のx値を求める(解の公式)
	//aが0のときにも対応。虚数のときは空の配列を返す
	private double[] calcXByF(int a, int b, int c) {
		if(a == 0){
			return new double[]{-1.0 * c / b};
		}
		
		int d = b * b - 4 * a * c;
		if(d < 0){
			return new double[]{};
		}
		else if(d == 0){
			return new double[]{(-1 * b) / (2 * a)};
		}
		else{
			double sqrt = Math.sqrt(d);
			double res1 = (-1 * b + sqrt) / (2 * a);
			double res2 = (-1 * b - sqrt) / (2 * a);
			return new double []{res1, res2};
		}
	}
	
	//2次関数のfx値を求める
	private double calcFx(int [] f, double resx) {
		double res = f[0] * resx * resx + f[1] * resx + f[2];
		return res;
	}
	
	//======幾何応用=====
	
	//凸包
	//生成後は反時計回りに格納されている O(n log n)
	//nが2のときは問題ない。nが1のときは空の配列が返される
	private Point2D [] convexHull(Point2D [] plist) {
		int n = plist.length;
		Arrays.sort(plist, com);
		int k = 0;
		Point2D [] qs = new Point2D[n * 2];
		for(int i = 0; i < n; i++){
			while(k > 1 && new Line2D.Double(qs[k-2] , qs[k-1]).relativeCCW(plist[i]) > 0){
				k--;
			}
			qs[k++] = plist[i];
		}
		for(int i = n - 2, t = k; i >= 0; i--){
			while(k > t && new Line2D.Double(qs[k-2] , qs[k-1]).relativeCCW(plist[i]) > 0){
				k--;
			}
			qs[k++] = plist[i];
		}
		Point2D [] res = Arrays.copyOf(qs, k-1);
		return res;
	}
	
	//ポリゴンカット
	//切り取った線の左側の多角形が残る。格納順に注意。反時計回りの多角形。p1からp2に向かう線分
	//O(n)
	private ArrayList<Point2D> polygonCut(ArrayList<Point2D> plist, Line2D cut) {
		int n = plist.size();
		ArrayList<Point2D> ans = new ArrayList<Point2D>();
		for(int i =0; i<n; i++){
			Point2D from = plist.get(i), to = plist.get((i+1)%n);
			if(cut.relativeCCW(from) <= 0){
				ans.add(from);
			}
			int temp1 = cut.relativeCCW(from);
			int temp2 = cut.relativeCCW(to);
			if(temp1 * temp2 < 0){
				Point2D IntersectP = intersectPtSS(cut, new Line2D.Double(from,to));
				ans.add(IntersectP);
			}
		}
		return ans;
	}
	
	//垂直二等分線
	//直線を生成する。値域が広いとき、線分の座標が小数のときは使えない
	private Line2D perpendicularBisector(Line2D l) {
		final long range = 10000;   //値域による
		Point2D lVec = sub(l.getP2(), l.getP1());
		Point2D midp = new Point2D.Double((l.getX1() + l.getX2()) / 2, (l.getY1() + l.getY2()) / 2);
		Point2D normalV = normalVector1(lVec);
		double tempx = normalV.getX() * range + midp.getX();
		double tempy = normalV.getY() * range + midp.getY();
		Point2D p1 = new Point2D.Double(tempx, tempy);
		double tempx2 = normalV.getX() * -range + midp.getX();
		double tempy2 = normalV.getY() * -range + midp.getY();
		Point2D p2 = new Point2D.Double(tempx2, tempy2);
		return new Line2D.Double(p2, p1);
	}
	
	//座標が同じかどうか
	private boolean issame(Point2D p1, Point2D p2) {
		if(Math.abs(p1.getX() - p2.getX()) < EPS && Math.abs(p1.getY() - p2.getY()) < EPS ){
			return true;
		}
		return false;
	}

	//線分アレンジメント
	//途中点の頂点は入っている。平行な線分同士の場合注意が必要
	class Sa{
		int INF = 1 << 24;
		class C implements Comparable<C>{
			int ind;
			Point2D p;
			public C(int ind, Point2D p) {
				this.ind = ind;
				this.p = p;
			}
			@Override
			public int compareTo(C o) {
				if(this.p.getX() < o.p.getX()) return -1;
				else if(this.p.getX() > o.p.getX()) return 1;
				else if(this.p.getY() < o.p.getY()) return -1;
				else if(this.p.getY() > o.p.getY()) return 1;
				else return 0;
			}
		}
		private double[][] segmentArrangement(Line2D[] line) {
			//交点リストを求める
			ArrayList<Point2D> intersectionlist = new ArrayList<Point2D>();
			for(int i = 0; i < line.length; i++){
				//intersectionlist.add(line[i].getP1());
				//intersectionlist.add(line[i].getP2());
				for(int j = i + 1; j < line.length; j++){
					if(line[i].intersectsLine(line[j])){
						intersectionlist.add(intersectPtSS(line[i], line[j]));
					}
				}
			}

			//sortする。重複削除
			Collections.sort(intersectionlist, com);
			for(int i = 1; i < intersectionlist.size(); i++){
				if(issame(intersectionlist.get(i-1), intersectionlist.get(i))){
					intersectionlist.remove(i);
					i--;
				}
			}

			//交点リストから、線分を通過するもの同士でグラフを生成する。
			int len = intersectionlist.size();
			double [][] res = new double[len][len];
			for(int i = 0 ; i< len; i++){
				Arrays.fill(res[i], INF);
				res[i][i] = 0.0;
			}
			for(int i = 0; i < line.length; i++){
				ArrayList<C> list = new ArrayList<C>();
				for(int j = 0; j < len; j++){
					if(line[i].ptSegDist(intersectionlist.get(j)) < EPS){
						list.add(new C(j, intersectionlist.get(j)));
					}
				}
				Collections.sort(list);
				for(int j = 1; j < list.size(); j++){
					int from = list.get(j-1).ind;
					int to = list.get(j).ind;
					res[from][to] = list.get(j).p.distance(list.get(j-1).p);
					res[to][from] = list.get(j).p.distance(list.get(j-1).p);
				}
			}
			return res;
		}
	}

	
	//=========3次元幾何=============
	
	//3次元の点
	class P3D{
		double [] p;
		public P3D(double[] p) {
			this.p = new double[3];
			for(int i =0; i < 3; i++){
				this.p[i] = p[i];
			}
			this.p = p;
		}
	}

	//3次元の直線
	class Line3D{
		P3D p1, p2;
		public Line3D(P3D p1, P3D p2) {
			this.p1 = p1;this.p2 = p2;
		}
	}

	//距離を求める。
	private double distance3D(P3D p1, P3D p2) {
		double res = 0;
		for(int i = 0; i < 3; i++){
			res += (p1.p[i] - p2.p[i]) * (p1.p[i] - p2.p[i]);
		}
		return Math.sqrt(res);
	}

	//ベクトルの距離を求める。normの平方根の値
	private double abs3D(P3D p) {
		return Math.sqrt(norm3D(p));
	}

	//ベクトルの値を求める。
	private double norm3D(P3D p) {
		return dot3D(p,p);
	}

	//内積を求める
	private double dot3D(P3D p1, P3D p2) {
		double res = 0.0;
		for(int i = 0; i < 3; i++){
			res += p1.p[i] * p2.p[i];
		}
		return res;
	}

	//足し算
	private P3D add3D(P3D p1, P3D p2) {
		double [] res = new double[3];
		for(int i = 0; i < 3; i++){
			res[i] = p1.p[i] + p2.p[i];
		}
		return new P3D(res);
	}
	
	//引き算。差分のベクトルを求める
	private P3D sub3D(P3D p1, P3D p2) {
		double [] res = new double[3];
		for(int i = 0; i < 3; i++){
			res[i] = p1.p[i] - p2.p[i];
		}
		return new P3D(res);
	}

	//かけ算。スカラー倍
	private P3D mul3D(double t, P3D p) {
		double [] res = new double[3];
		for(int i = 0; i < 3; i++){
			res[i] = p.p[i] * t;
		}
		return new P3D(res);
	}
	
	//角度を求める
	private double angle3D(P3D p1, P3D p2) {
		double a = dot3D(p1,p2);
		double b = abs3D(p1);
		double c = abs3D(p2);
		return Math.acos(a / (b * c));
	}

	//射影。直線l上にある、点pを垂直におろした点を返す。
	private P3D projection3D(Line3D l, P3D p) {
		double t = dot3D(sub3D(p, l.p1), sub3D(l.p1, l.p2)) / norm3D(sub3D(l.p1, l.p2));
		return add3D(l.p1, mul3D(t, (sub3D(l.p1, l.p2))));
	}

	//反射。直線lを軸としてpの反射位置にある点を返す。
	private P3D reflection3D(Line3D l, P3D p) {
		return add3D(p, mul3D(2.0, sub3D(projection3D(l,p) , p)));
	}

	//点と直線の距離
	private double distanceLP3D(Line3D l, P3D p) {
		return abs3D(sub3D(p, projection3D(l,p)));
	}
	
//==========not Library ?====================
	
	//多角形の辺をmidの値分内側に平行移動させた線分を返す。
	private Line2D getcutV(Point2D p1, Point2D p2, double mid) {
		Point2D p2p1 = sub(p2, p1);
		double p1p2Dis = p1.distance(p2);
		//内側の法線ベクトル。反時計回りなので、１つにしぼられる
		Point2D p2p1NV = normalVector1(p2p1);
		Point2D cutV = mul(mid / p1p2Dis, p2p1NV);
		Point2D cutp1 = add(p1, cutV);
		Point2D cutp2 = add(p2, cutV);
		Line2D res = new Line2D.Double(cutp1, cutp2);
		return res;
	}
	
	//半径1の円周上の２点から中心点を求める
	private Point2D[] centerPoint(Point2D p1, Point2D p2) {
		double r = 1.0;
		double midx = (p1.getX() + p2.getX()) / 2.0;
		double midy = (p1.getY() + p2.getY()) / 2.0;
		Point2D midP = new Point2D.Double(midx, midy);
		Point2D p2p1 = sub(p2,p1);
		Point2D normalV1 = normalVector1(p2p1);
		Point2D normalV2 = normalVector2(p2p1);
		double halfdis = p1.distance(p2) / 2.0;
		if(halfdis > r + EPS) return null;
		double tocenterDis = Math.sqrt(r - halfdis * halfdis);
		Point2D [] res = new Point2D[2];
		res[0] = add(mul(tocenterDis / abs(normalV1), normalV1), midP);
		res[1] = add(mul(tocenterDis / abs(normalV1), normalV2), midP);
		return res;
	}
	
	//y座標からxを計算する関数
	//a[i][0]はx座標、a[i][1]はy座標を入れている
	//i番目の座標とi+1番目の座標からなる線分について、y座標yの値のときのxを求める。
	//切り上げした値と切り下げした値を返す
	private int[] computeX(int i, int y, int [][] a) {
		int n = a.length;
		int [] res = new int[2];
		int den, num, w;
		int nextind = (i + 1) % n;
		den = a[nextind][1] - a[i][1];
		num = (y - a[i][1]) * (a[nextind][0] - a[i][0]);
		w = a[i][0] + num / den;
		if(num % den == 0){
			res[0] = w;
			res[1] = w;
		}
		else if((num % den) * den < 0){
			res[0] = w-1;
			res[1] = w;
		}
		else{
			res[0] = w;
			res[1] = w + 1;
		}
		return res;
	}
}
