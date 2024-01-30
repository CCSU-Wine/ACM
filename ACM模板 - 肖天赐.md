



<div align = "center" style = 'font-size:65px'><b>ACM算法模板<b></div>



![](C:\Users\饕餮\Pictures\Saved Pictures\acm2.jpg)





<div style = 'font-size:35px'><b>长沙学院-肖天赐-2023<b></div>

















[TOC]













































# 开题策略

- 前期跟榜的同时，要自主开题。
- 前期简单签到题不着急交，多一两分钟罚时换避免wa一发策略更优，可以三个人一起签到。
- 读完题要确认按意思是否能通过样例（读题一人读完，第二人需要复查一遍，看有无遗漏），对于想到的思路，检查是否能通过题目所给的样例！
- 赛中注意跟榜，不要死磕一题。
- 坚持到最后一刻，当陷入僵局时根据榜上局势勇于开题。



# 易错检查

- 注意题目给定的时间空间复杂度。 

- 检查数组有无越界，链式前向星存边需要开 $2$ 倍，线段树 $4$ 倍。检查STL容器有无为空时进行访问。

- 检查答案是否需要 long long 或者 __int128，是否需要取模，当模数较大时是否会乘爆 long long.

- 检查答案是否允许行末空格和尾部换行。

- 有无特殊样例，$0,1$ 等。

- 当遇到 map 超时时，考虑换成 unordered_map 或 pbds 库中的哈希表。

- 线段树检查查询修改区间判断是否正确，懒标记是否下传，清空。当同时有清空懒标记和 laz 时，应该先后都 pushdown.

- 重载运算符时，优先队列的重载与普通数组相反。

  

# 竞赛基础



## 编译

DevC++ 开大栈空间

Dev_C++中 工具 → 编译选项，在编译时加入以下指令处打上勾，同时加入以下代码。

```C++
-Wl,-stack=134217728
```

$134217728 = 128∗1024∗1024$，即 $128MB$ 的空间。



手动开栈  $512M$ 适用于没有无限栈的评测机

```C++
int main() {
    int size(512<<20); // 512M
    __asm__ ( "movq %0, %%rsp\n"::"r"((char*)malloc(size)+size)); 
    /*
        YOUR CODE
        ...
    */
    exit(0); // 必须以此结尾
}
```



手动开O(2).

```C++
#pragma GCC optimize(2)
#pragma GCC optimize(3,"Ofast","inline")
```



## 快读快写

```C++
//快读
inline int read()
{
	int x = 0, y = 1;char c = getchar();
	while (c < '0' || c>'9') { if (c == '-') y = -1;c = getchar(); }
	while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
	return x * y;
}
//快写
inline void write(int x)
{
    if(x<0) putchar('-'),x=-x;
    if(x>9) write(x/10); putchar(x%10+'0');
}
//加速cin,cout
int main()
{
    //加速cin,cout 不能和scanf，printf同时使用
    ios::sync_with_stdio(false);
	cin.tie(nullptr);cout.tie(nullptr);
}
```



## __int128输入输出

```C++
inline __int128 scan()
{
    __int128 x=0,f=1;
    char ch=getchar();
    while(ch<'0'||ch>'9'){
        if(ch=='-')
            f=-1;
        ch=getchar();
    }
    while(ch>='0'&&ch<='9'){
         x=x*10+ch-'0';
         ch=getchar();
    }
    return x*f;
}
void print(__int128 x)
{
    if(x < 0)
    {
        x = -x;
        putchar('-');
    }
    if(x > 9) print(x/10);
    putchar(x%10 + '0');
}
```



## 函数

```C++
assert(siz[1] == m);//当为false时Runtime error,可以用在怀疑有问题但不确定的情况下
```

```C++
//取整函数
floor(x);//向下取整
ceil(x); //向上取整
round(x);//四舍五入
fix(x);  //朝0方向取整

cout << fixed << setprecision(x) << num; // 保留x位小数输出

// if+while+位运算（超快） 此段代码a、b可以为0
inline int gcd(int a,int b) {    
    if(b) while((a%=b) && (b%=a));    
    return a+b;
}

// 全排列函数
while(prev_permutation(a + 1, a + 1 + n)); // 向前
while(next_permutation(a + 1, a + 1 + n)); // 向后

__builtin_popcount(x); // int范围内 x二进制1的个数
__builtin_popcountll(x); // long long 范围内 x二进制1的个数
```



## 对拍

```C++
Match.cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    int t = 1000;
    while(t--) {
        system("rand.exe > data.in");
        system("AC.exe < data.in > AC.out");
        double st = clock();
        system("WA.exe < data.in > WA.out");
        double ed = clock();
        if(system("fc AC.out WA.out")) {
             cout << "Wrong Anwser" << endl;
            break;
        }
        else cout << "Accpeted -- Case: " << t << " 用时 : " << ed - st << "ms "<< endl;
    }
    return 0;
}  

rand.cpp//随机生成数据
srand((unsigned)time(0));
rand() % mod;// 0 ~ mod的数

mt19937_64 mrand(random_device{}());//随机生成大数
mrand() % mod;


AC.cpp//正确代码，暴力
AC.out//正确答案
   
WA.cpp//待测代码
WA.out
    
data.in//输入数据   
```



### 随机生成数据

#### 生成树

```C++
// 生成树
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>

using namespace std;
int a[100000];

int random(int n){
	return (long long) rand() * rand() % n;
}

int main(){
	srand((unsigned)time(0));
	int n, m;
	scanf("%d", &n);
	for (int i = 2; i <= n; i ++ ){
		int fa = random(i - 1) + 1;
		int val = random(1000000) + 1;
		printf("%d %d %d\n", fa, i, val);
	}
}
```



#### 生成无重边与自环的图

```C++
//无重边与自环的图
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <map>
using namespace std;
// 5 <= n <= m <= n * (n - 1)/4 <= 1e6
pair<int, int> e[1000005];  // 保存数据
map<pair<int, int>, bool> h; // 防止重边
int random(int n)
{ 	
	return (long long) rand() * rand() % n;
}

int main()
{
	srand((unsigned)time(0));
	int n, m;
	cin >> n >> m;
	// 先生成一棵树, 保证连通
	for (int i = 1; i < n; i ++ ) {
		int fa = random(i) + 1;
		e[i] = make_pair(fa, i + 1);
		h[e[i]] = h[make_pair(i + 1, fa)] = 1;
	}
	
	//再生成剩余的m-n+1条边
	
	for (int i = n; i <= m; i ++ ) {
		int x, y;
		do {
			x = random(n) + 1, y = random(n) + 1;
		}while (x == y || h[make_pair(x, y)]);
		
		e[i] = make_pair(x, y);
		
		h[e[i]] = h[make_pair(y, x)] = 1;
	}
	random_shuffle(e + 1, e + m + 1);
	for (int i = 1; i <= m; i ++ )
		printf("%d %d\n", e[i].first, e[i].second);
	return 0;
}
```



#### 生成全排列

```C++
// O(N)随机生成全排列
#include <bits/stdc++.h>
using namespace std;
int main()
{
    srand((unsigned)time(0));
    int n = 400;
    cout << n << "\n";
    vector<int>a, b;
    a.push_back(0);
    for(int i = 1; i <= n; i ++) a.push_back(i);
    int t = n;
    for(int i = 1; i <= n; i ++){
        int x = 1 + rand() % t;
        b.push_back(a[x]);
        swap(a[x], a[t --]);
    }
    for(int i = 0; i < n; i ++) cout << b[i] << " ";
    return 0;
}
```



## STL相关

### 重载

```C++
/*重载pair 运算符*/
//pair相加
template<class Ty1,class Ty2> 
inline const pair<Ty1,Ty2> operator+(const pair<Ty1, Ty2>&p1, const pair<Ty1, Ty2>&p2)
{
    pair<Ty1, Ty2> ret;
    ret.first = p1.first + p2.first;
    ret.second = p1.second + p2.second;
    return ret;
}

//hash_map重载以pair为key值
struct hash_pair { 
    template <class T1, class T2> 
    size_t operator()(const pair<T1, T2>& p) const
    { 
        auto hash1 = hash<T1>{}(p.first); 
        auto hash2 = hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
}; 
unordered_map<pair<int, bool>, int, hash_pair> f;

/* set/map/priority_queue 重载自定义类型*/ 
struct point{
    ll x, y;
};
struct cmp{
    bool operator() (const point &A,const point &B){
        return  A.y * B.x > A.x * B.y; //按斜率比较
    }
};
set<point, cmp>s;
map<point, Ty, cmp>mp;

/*优先队列重载*/
struct node{
    int u, d;
    bool operator < (const node &A)const{
        //对于优先队列需要反一下符号
        return d < A.d;//大顶堆
        return d > A.d;//小顶堆
    }
};
priority_queue<node>q;
//当使用默认类型时，不重载默认为大根堆
priority_queue<int>q;//大根堆
priority_queue<int, vector<int>, greater<int> >q;//小根堆
```

### 时间复杂度

```C++
/* set */
multiset
注意count函数时间复杂度为 O(k + logn) 其中n为集合中数的数量，k为查询的数的数量
```

### 简单应用

```C++
/* set */
//迭代器 next(it, i)指向后i位迭代器 prev(it, i)指向前i位迭代器 map同样可以
auto it = r.lower_bound(24);//it指向24所在
printf("*next(it) = %d\n",*next(it));//默认为后一位
printf("*next(it,2) = %d\n",*next(it,2));

auto it1 = r.lower_bound(99);//it1指向100所在
printf("*prev(it) = %d\n",*prev(it1));
printf("*prev(it, 4) = %d\n",*prev(it1, 4));

/* vector */
vector < int > v(n, data);//一维vector初始化
vector< vector<int> > V(n, vector<int>(m, data));//二维vector初始化 n行m列 均是从0开始，一般初始化为n + 1和m + 1, date为赋初值

/* string */
```



## pbds库

头文件与命名空间如下

```C++
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_cxx;
using namespace __gnu_pbds;
```

### 平衡树

声明如下

```C++
tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update>
```

第一个参数：键（key）的类型，`int`

第二个参数：值（value）的类型，`null_type` 表示无映射，表明这是 set 不是 map.在一些较低版本中写为 `null_mapped_type`

第三个参数：仿函数类，表示比较规则，`less<int>`

第四个参数：平衡树的类型

- `rb_tree_tag`（红黑树）一般使用这种速度较快

- `splay_tree_tag`（伸展树）
- `ov_tree_tag` （有序向量树）

第五个参数：元素维护策略，`tree_order_statistics_node_update` 才可以维护 $k_{th}$ 和 $rank$，默认值为`null_tree_node_update`.

基本操作：

```C++
insert(x); // 插入x
erase(x); // 删除x

find_by_order(k);/* 求平衡树内排名为k的值是多少
返回为迭代器 it，其中k的值域为[0, size-1]，若不在这个范围内，直接返回end()	*/   

order_of_key(x); /* 求x的排名
返回的为整数，此处x不一定要存在于平衡树中
该方法实现原理是求严格小于x的元素个数，最终我们需要加上1才是答案 */

lower_bound(x); /* 求大于等于x的最小值
返回的为迭代器，没找到时返回 end() */

upper_bound(x); /* 与lower_bound(x)同*/

join(b); /* 合并，要求合并的两颗树的 key 值没有相同的，合并后 b 树被清空 */

split(v, b); /* v是一个key值类型，将小于等于v的元素归属于a，其余元素归属于b，b中原来元素将会被清空 */
 
```



例题：维护以下操作

1. 插入 $x$ 

2. 删除 $x$ (若有多个相同的数，只删除一个)
3. 查询 $x$ 的排名(排名定义为比当前数小的数的个数 $+1$)
4. 查询排名为 $x$ 的数
5. 求 $x$ 的前驱(前驱定义为小于 $x$，且最大的数)
6. 求 $x$ 的后继(后继定义为大于 $x$，且最小的数)

由于 `pbds` 中的平衡树不能有重复的值，对于重复的值我们可以用 `pair<int,int>` 来表示

```C++
#include <bits/stdc++.h>
#include <bits/extc++.h>
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;

typedef pair<int, int> pii;
typedef tree<pii, null_type, less<pii>, rb_tree_tag, tree_order_statistics_node_update> Tree;

const int inf = 1e9;

Tree tr;
int tr_cnt;

template<typename T> int get_order(T x){ // 获取x的排名
    return tr.order_of_key(x) + 1;
}

template<typename T> int get_key(T k){ // 获取排名为k的键
    auto it = tr.find_by_order(k - 1);
    if(it == tr.end()) return inf;
    return it->first; 
}

template<typename T> void erase_key(T x){
    auto it = tr.lower_bound(x); // 找到任意该数后删除迭代器
    if(it->first == x.first) tr.erase(it);
}

template<typename T> int get_pre(T x){ // 获取前驱
    auto it = tr.lower_bound(x);
    if(it == tr.begin()) return -inf; 
    return prev(it)->first;
}

template<typename T> int get_sub(T x){ // 获取后继
    auto it = tr.upper_bound(x);
    if(it == tr.end()) return inf;
    return it->first;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int n;
    cin >> n;
    for(int i = 1; i <= n; i ++){
        int op, x;
        cin >> op >> x;
        if(op == 1) tr.insert(make_pair(x, ++ tr_cnt));
        else if(op == 2) erase_key(make_pair(x, 0));
        else if(op == 3) cout << get_order(make_pair(x, 0)) << "\n";
        else if(op == 4) cout << get_key(x) << "\n";
        else if(op == 5) cout << get_pre(make_pair(x, 0)) << "\n";
        else if(op == 6) cout << get_sub(make_pair(x, inf)) << "\n";
    }
    return 0;
}
```




# 基础算法

## 龟速乘

```C++
/* 防止模数过大导致整型溢出 */
#define ll long long
ll qmul(ll a, ll b, ll p)
{
    ll res = 0;
    while(b){
        if(b & 1) res = (res + a) % p;
        a = (a + a) % p;
        b >>= 1;
    }
    return res % p;
}
```



## 三分法求单峰函数极值

​		三分一般不适用于表达式中有取整的函数， 可能会造成有多段值相同的区间，导致端点收缩过程判断有误。

```C++
#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-8;
int n;   
double p[20];
double getres(double x){
    double res = 0;
    for(int i = n; ~i; i --){
        res = res * x + p[i];
    }
    return res;
}
int main()
{
    double l,r; 
    cin >> n >> l >> r;
    for(int i = n; ~i; i --) {
        cin >> p[i];
    }
    for(int i = 1; i <= 100; i ++) {
        double midl = (l * 2 + r) / 3;
        double midr = (r * 2 + l) / 3;
        if(getres(midl) < getres(midr)) l = midl;
        else r = midr;
    }
    printf("%.8lf",l);
    return 0;
}

//整数三分，用于答案必须取整的题目
void solve()
{
    scanf("%lf%lf%lf",&n,&m,&p);
    p = p / 10000.0;
    ll l = 1, r = 1e6;
    while(l < r - 1){
        ll mid = (l + r) >> 1;
        ll midl = mid - 1, midr = mid + 1;
        double resl = getExc(midl), resr = getExc(midr);
        if(resl > resr) {
            if(l == midl) l = mid;
            else l = midl;
        }
        else {
            if(r == midr) r = mid;
            else r = midr;
        }
    }
    printf("%.8lf\n",min(getExc(l),getExc(r)));
    
    //三分实数，对答案取整取最值
    printf("%.8lf\n",min(getExc(ceil(l)), getExc(floor(l))));
    return ;
}
```



## 高精度

### 高精度加法

```C++
typedef vector<int> Vec;
Vec add(Vec &A, Vec &B)
{
    Vec C;
    int t = 0;
    for(int i = 0; i < A.size() || i < B.size(); i ++){
        if(i < A.size()) t += A[i];
        if(i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if(t) C.push_back(1);
    return C;
}
int main()
{
    ios::sync_with_stdio(false);
	cin.tie(nullptr);cout.tie(nullptr);

    string a, b;
    Vec A, B, ans;
    cin >> a >> b;
    for(int i = a.length() - 1; i >= 0; i --) A.push_back(a[i] - '0');
    for(int i = b.length() - 1; i >= 0; i --) B.push_back(b[i] - '0');

    ans = add(A, B);

    for(int i = ans.size() - 1; i >= 0; i --) cout << ans[i];
    return 0;
}
```



### 高精度减法

```C++
typedef vector<int>Vec;
bool cmp(Vec &A, Vec &B) // 判断A >= B
{
    if(A.size() != B.size()) return A.size() > B.size();
    for(int i = A.size() - 1; i >= 0; i --){
        if(A[i] != B[i]) return A[i] > B[i];
    }
    return true;
} 
Vec sub(Vec &A, Vec &B){
    Vec C;
    for(int i = 0, t = 0; i < A.size(); i ++){
        t = A[i] - t;
        if(i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if(t < 0) t = 1;
        else t = 0;
    }
    while(C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
int main()
{
    ios::sync_with_stdio(false);
	cin.tie(nullptr);cout.tie(nullptr);

    string a, b;
    Vec A, B, ans;
    cin >> a >> b;
    for(int i = a.length() - 1; i >= 0; i --) A.push_back(a[i] - '0');
    for(int i = b.length() - 1; i >= 0; i --) B.push_back(b[i] - '0');

    if(cmp(A, B)) ans = sub(A, B);
    else {
        ans = sub(B, A); 
        cout << "-"; 
    }
    for(int i = ans.size() - 1; i >= 0; i --) cout << ans[i];
    return 0;
}
```



### 高精度乘法

```C++
typedef vector<int>Vec;
Vec mul(Vec &A, int B)
{
    Vec C;
    if(!B) return C = {0};
    for(int  i = 0, t = 0; i < A.size() || t; i ++){
        t += A[i] * B;
        C.push_back(t % 10);
        t /= 10;
    }
    return C;
}
int main()
{
    ios::sync_with_stdio(false);
	cin.tie(nullptr);cout.tie(nullptr);

    int B;
    string a;
    cin >> a >> B;
    Vec A,ans;
    for(int i = a.length() - 1; i >= 0; i --) A.push_back(a[i] - '0');
    ans = mul(A, B);
    for(int i = ans.size() - 1; i >= 0; i --) cout << ans[i];
    return 0;
}
```



### 高精度除法

```C++
typedef vector<int> Vec;
Vec div(Vec &A ,int B,int &r) //r返回余数
{
    Vec C;
    for(int i = A.size() - 1; i >= 0; i --){
        r = r * 10 + A[i];
        C.push_back(r / B);
        r %= B;
    }
    reverse(C.begin(), C.end());
    while(C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
int main()
{
    ios::sync_with_stdio(false);
	cin.tie(nullptr);cout.tie(nullptr);

    int B, r = 0;
    string a;
    Vec A, ans;
    cin >> a >> B;
    
    for(int i = a.length() - 1; i >= 0; i --) A.push_back(a[i] - '0');
    ans = div(A, B, r);
    for(int i = ans.size() - 1; i >= 0; i --) cout << ans[i];
    cout << "\n" << r;
    return 0;
}
```



# 数论



## 结论和猜想

### 哥德巴赫猜想以及推论

基本猜想：任一大于 $5$ 的整数都可写成三个质数之和。

等价猜想：任一大于 $2$ 的偶数都可写成两个质数之和。

弱猜想：任一大于 $7$ 的奇数都能被表示成三个奇质数的和。



### 因子个数

一个整数可以写成质因子的幂次方相乘的形式。通式如下，其中 $p$ 为质数。

​								$N = {p_1}^{a_1} * {p_2}^{a_2} * {p_3}^{a_3} * \cdots * {p_n}^{a_n}$

而其因数个数 

​								$sum = (a_1 + 1) * (a_2 + 1) * \cdots * (a_n + 1)$

​		原理：对于一个数 $N$ 选择其质因子的个数或者种数不同相乘得到的因子也不同。对于每个质因子的选法，有 $0 \sim a_i$ 个共 $a_i+1$ 种可能，所有质因子的选法相乘即为结果。

```C++
//求解n个数相乘的因子数量
#include<map>
#include<iostream>
#include<algorithm>
using namespace std;
#define ll long long
const int mod = 1e9 + 7;
map<int ,int>mp;//记录所有数数质因子个数
int main()
{
    int n, x;
    scanf("%d",&n);
    for(int i = 1; i <= n; i ++){
        scanf("%d",&x);
        for(int j = 2; j <= x / j; j ++){
            int cnt = 0;
            while(x % j == 0){
                cnt ++;
                x /= j;
            }
            if(cnt) mp[j] += cnt;
        }
        if(x > 1) mp[x] ++;
    }
    ll ans = 1;
    for(auto it : mp){
        ans = ans * (it.second + 1) % mod;
    }
    printf("%lld",ans);
    return 0;
}
```



### GCD与LCM

1. $\gcd(a, b) <= a - b$ （假设 $a > b$）

2. 已知 $a$, $b$，且 $x + y = a$ ，$lcm(x,y) = b$，则有 $\gcd(a, b) = \gcd(x, y)$
3. $\gcd(a, b) = 1$ 可以得出 $\gcd(a + b, ab) = 1$

#### 斐蜀定理

设 $a,b$ 是不全为 $0$ 的整数，则存在整数 $x,y$ 使得 $ax+by=\gcd(a,b)$.



## 互质与欧拉函数

### 欧拉函数

定义：对于一个正整数 $n$，欧拉函数 $\varphi(n)$ 表示小于等于 $n$ 与 $n$ 互质的正整数个数。

性质：

1. **如果 $p$ 是质数，$\varphi(p) = p - 1$**，即只与自己本身不互质。

   ​					**$\varphi(p^n) = p^{n-1}*(p-1)$**

   即只有$p$ 的倍数： $p,2p,\dots,p^{n-1}*p$ 与之不互质共 $p^{n-1}$ 个：

   ​					$\varphi(p^n) = p^n - p^{n-1} = p^{n - 1}*(p-1)$

   

2. **如果 $p$，$q$ 互质，$\varphi(p * q) = \varphi(p) * \varphi(q) = (p - 1) * (q - 1)$**.

   推导：$p,2p,3p,\dots,(q - 1) * p$ 有 $q - 1$ 个数整除 $p * q$.

   同理：$q,2q,3q,\dots,(p - 1) * q$ 有 $p - 1$ 个数整除 $p * q$.

   **因为 $p,q$ 互质，不存在任意小于 $q$ 的正整数 $x$ 和 小于 $p$ 的正整数 $y$，使得 $xp = yq$.**

   所以上述的所有数字都是不同的可以整除 $p,q$ 的正整数，最后加上 $p * q$ 本身，剩下的数就是与之互质的。

   综上：$\varphi(p * q) = p * q - (p - 1) - (q - 1) - 1 = (p - 1) * (q - 1)$

   

3. **如果 $a\mid x$，则有 $\varphi(ax) = a\varphi(x)$**.



欧拉函数的求法：

我们将一个正整数 $n$ 质因数分解：$n = p_1^{k_1}p_2^{k_2}\dots p_m^{k_m}$，根据性质 1，2有

$\varphi(n) = p_1^{k1-1}(p_1-1)*p_2^{k2-1}(p_2-1)*\dots*p_m^{k_m-1}(p_m-1)$ 

$\varphi(n) = p_1^{k_1}\frac {p_1-1}{p_1} * p_2^{k_2}\frac {p_2-1}{p_2} * \dots * p_m^{k_m}\frac {p_m-1}{p_m}$，而 $\prod_{i=1}^m p_i^{k_i} = n$

$\varphi(n) = n * \frac {p_1-1}{p_1} *\frac {p_2-1}{p_2} * \dots * \frac {p_m-1}{p_m}$

 

#### 求指定欧拉函数

​		求单个指定正整数 $n$ 的欧拉函数，最坏时间复杂度为 $O(\sqrt{n})$，即分解质因数的复杂度。

```C++
#include <bits/stdc++.h>
using namespace std;

int phi(int n){
    int res = n;
    for(int i = 2; i <= n / i; i ++){
        if(n % i == 0) res = res / i * (i - 1);// 先除再乘防止溢出
        while(n % i == 0) n /= i;
    }
    if(n > 1) res = res / n * (n - 1); // 本身是质数
    return res;
}

int main(){
    int t;
    cin >> t;
    while(t --){
        int n; cin >> n;
        cout << phi(n) << "\n";
    }
    return 0;
}
```



#### 筛法求欧拉函数

​		由欧拉公式 $\varphi(n) = n * \frac {p_1-1}{p_1} *\frac {p_2-1}{p_2} * \dots * \frac {p_m-1}{p_m}$ 可知，一个数的欧拉函数仅与其质因数的种类有关，与数量幂次无关，而筛法恰好就是利用质数去筛出合数保留质数的过程。

​		求多个正整数欧拉函数，可以与筛法结合。求 $1 \sim n$ 的欧拉函数，埃氏筛时间复杂度为 $O(nloglogn)$，欧拉筛时间复杂度近似于 $O(n)$.

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 1e6 + 10, MAX = 1e6;

int prim[N], phi[N], vis[N], cnt;

//埃氏筛
void get_phi(){
    vis[1] = 1;
    for(int i = 1; i <= MAX; i ++) phi[i] = i;
    for(int i = 1; i <= MAX; i ++){
        if(vis[i]) continue ;
        for(int j = 1; j * i <= MAX; j ++){
            vis[i * j] = 1;
            phi[i * j] = phi[i * j] / i * (i - 1); // i为筛出的合数的质因子
        }
        vis[i] = 0;
    }
}

//欧拉筛
void get_phi(){
    vis[1] = phi[1] = 1;
    for(int i = 2; i <= MAX; i ++){
        if(!vis[i]){
            phi[i] = i - 1;
            prim[++ cnt] = i;
        }
        for(int j = 1; prim[j] <= MAX / i; j ++) {
            vis[prim[j] * i] = 1;
            if(i % prim[j] == 0) { // 说明i 与 prim[j] * i 的质因数种类相同
                phi[prim[j] * i] = phi[i] * prim[j]; // 由性质3可得
                break;
            }
            phi[prim[j] * i] = phi[i] * (prim[j] - 1); // 说明 i 与 prim[j] 互质，由性质2可得
        }
    }
}
```



### 欧拉定理/费马小定理

欧拉定理：若 $a,n$ 互质 $a^{\varphi(n)} \equiv 1(\mod n)$.

费马小定理：若 $p$ 是质数，则对于任意整数 $a$，有 $a^p \equiv a(\mod p)$.

其中 **欧拉函数φ(n)** 是小于或等于 $n$ 的正整数中与 $n$ 互质的数的数目。



### 欧拉降幂（推论）

当计算的是指数级别的幂次，使用快速幂显然无法求解如此高的幂次，于是我们用到欧拉降幂。

**若正整数 $a, n$ 互质 $(gcd(a, n) = 1)$，则对于任意正整数 $b$，有 $a^b \equiv a^{b\mod\varphi(n)}(\mod n)$**.

证明：

​		设 $b = q * \varphi(n) + r$，其中 $0 \leq r < \varphi(n)$

​		即 $r = b \mod \varphi(n)$

于是有：

​		$a^b \equiv a^{q*\varphi(n) + r} \equiv (a^{\varphi(n)})^q * a^r \equiv 1^q * a^r \equiv a^{b\mod\varphi(n)}(\mod n)$

1. **一般情况中， 模数一般为一个质数 $p$， 而质数的欧拉函数 $\varphi(p) = p - 1$，所以我们对指数取模 $p - 1$ 即可。**

2. **特别地，当 $a,n$ 不互质且 $b > \varphi(n)$ 时 **

    $a^b\equiv a^{b\mod\varphi(n)+\varphi(n)}(\mod n)$



### 欧拉筛

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e7+5;
const int MAX = 1e7;
int st[N],prim[N],cnt,n;//prim质数表
//欧拉筛，线性筛
void get_primes()//原理：所有合数只被其最小质因子筛一遍
{
    for(int i = 2; i <= n; i ++){
        if(!st[i]) prim[++ cnt] = i;//筛出质数放入质数表中
        for(int j = 1; prim[j] <= n/i; j ++){
            st[prim[j] * i] = 1;
            if(i % prim[j] == 0) break;//说明prim[j]是i的最小质因子
        }
    }
}
```



## 函数

### 函数极值

双勾函数极值：形如 $f(x) = ax + \frac bx$ 极值在 $x = \sqrt{\frac ba}$ 取到。



### 解一元二次方程

1. 判断一元二次方程 $ax^2+bx+c=0$ 是否有实数解，可以根据判别式 $D=b^2-4ac$ 的正负性进行判断。具体地：

   如果 $D>0$，则方程有两个不相等的实数解。

   如果 $D=0$，则方程有且仅有一个实数解。

   如果 $D<0$，则方程没有实数解，但有两个共轭复数解。

2. 一元二次方程 $ax^2+bx+c=0$ 的解为 $x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$. 为了判断该方程是否有整数解，我们需要确定 $\sqrt {b^2-4ac}$ 是否为整数。

    如果 $\sqrt{b^2-4ac}$ 是整数，则方程有整数解。否则，方程没有整数解。

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
typedef pair<ll, ll>pII;
int check(ll a, ll b, ll c){ // 判断是否有实数解
    ll D = b * b - a * c * 4;
    if(D < 0) return -1;
    if(D > 0) return 2;
    return 1;
}

ll checkint(ll a, ll b, ll c){ // 判断是否有正整数解
    if(check(a, b, c) == -1) return -1;
    ll D = b * b - a * c * 4;
    ll xi = sqrtl(b * b - 4 * a * c);
    for(ll i = max(0LL, xi - 5); i <= xi + 5; i ++){
        if(i * i == D) return i;
    }
    return -1;
}

pII solving(ll a, ll b, ll c){ // 形如ax^2 + bx + c = 0
    ll D = b * b - a * c * 4;
    ll v = checkint(a, b, c);
    ll res1 = (-b + v) / (2 * a);
    ll res2 = (-b - v) / (2 * a);
    return {res1, res2};
}
```



## 矩阵快速幂

```C++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int mod = 1e9 + 7, MAX = 110;

int n, k;
struct Matrix
{
    ll Mat[MAX][MAX];
	Matrix(){//构造方法，申请变量时自动赋值
        for(int i = 0; i < MAX; i ++)
            for(int j = 0; j < MAX; j ++) Mat[i][j] = 0;
    }
    inline void build(){
        for(int i = 0; i < MAX; i ++) Mat[i][i] = 1;//构造单位矩阵
    }
};
Matrix operator * (const Matrix &A, const Matrix &B)//重载乘法, siz为矩阵具体大小
{
    Matrix a;
    for(int i = 0; i < siz; i ++){
        for(int j = 0; j < siz; j ++){
            for(int k = 0; k < siz; k ++){
                a.Mat[i][j] += A.Mat[i][k] * B.Mat[k][j] % mod;
                a.Mat[i][j] %= mod;
            }
        }
    }
    return a;
}
Matrix ksm(Matrix a, ll b)
{
    Matrix ans;
    ans.build();
    while(b)
    {
        if(b & 1) ans = ans * a;
        a = a * a;
        b >>= 1;
    }
    return ans;
}
void solve()
{
    Matrix A,B;
    cin >> n >> k;
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j++){ 
            scanf("%lld",&A.Mat[i][j]);
        }
    }
    B = ksm(A,k);
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++) {
            printf("%lld ",B.Mat[i][j]);
        }
        puts("");
    }
}
```



## 排列组合



### 组合数学

#### 求小规模组合数

```C++
#include <bits/stdc++.h>
using namespace std;
__int128 C(int n, int m){
    __int128 ans = 1;
    int l = 1, r = n;
    m = min(m, n - m);
    while(l <= m){
        ans = ans * (r --) / (l ++);
    }
    return ans;
}
```

#### 递推求组合数

递推式：

​								$C_n^m = C_{n-1}^{m-1}+C_{n-1}^m$

证明：从 $n$ 中取出一个数确定，将方案分为两种

1. 包含取出的数的方案，即从剩下的 $n-1$ 个中取出 $m-1$ 个和提前取出的数组合成 $m$ 个。
2. 不包含取出的数的方案，即从剩下的 $n-1$ 个取出 $m$ 个。

```C++
int C[N][N];
void init(){
	for(int i = 0; i <= MAX; i ++){
        C[i][0] = 1;
		for(int j = 1; j <= i; j ++){
			C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % mod;
		}
	}
}
```



#### 阶乘逆元求组合数/卡特兰数

性质：一种操作的数量等于另一种操作，并且任意时刻 $1$ 号操作数量的前缀大于等于 $2$ 号操作。

经典问题：长度为 $2n$ 的合法括号序列。

​		证明：转化为走格子问题，从 $(0,0)$ 出发每次可以选择向上走一步或向右走一步，最终走到 $(n,n)$，$x + 1$ 视为向右走一步 ， $y + 1$ 视为向上一步，要求任意时刻在格子中坐标  $(i,j)$，$i\geq j$  问最终走到 $(n,n)$ 的满足要求的走法方案数。 

​		总方案数：从 $2n$ 个位置中任选 $n$ 个位置进行向右走的操作，剩下的自然就是向上走

​										$sum = C_{2n}^{n}$ 
​		不合法方案数：存在 $y > x$ 说明在走的过程中一定走到过 $y = x + 1$ 的线上，于是路径就能转化成走到 $(n, n)$ 以该线为对称轴的对称点 $(n-1,n+1)$ 的方案 

​										$sub = C_{2n}^{n-1}$
最终合法方案数：  

​										$ans = C_{2n}^{n} - C_{2n}^{n-1}$  

​		此类问题可以拓展到 $n$ 次 $1$ 类操作 $m$ 次 $2$ 类操作，要求前缀数目 $pre_1+k >= pre_2$ ，相应的对称轴变成 $y = x + k + 1$.

​										$ans = C_{n + m}^{n} - C_{n + m}^{n + k + 1}$

求组合数，时间复杂度为 $O(n\log_2m)$ 其中 $n$ 为阶乘大小， $m$ 为模数大小。

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 2e5 + 10, mod = 1e9 + 7; // 记得空间 * 2

ll ksm(ll a, ll b){
    ll res = 1;
    while(b){
        if(b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1; 
    }
    return res;
}

ll fact[N],infact[N];
void init(int n){
    fact[0] = infact[0] = 1;
    for(int i = 1; i <= n; i ++){
        fact[i] = fact[i - 1] * i % mod;
    }
    infact[n] = ksm(fact[n], mod - 2);
    for(int i = n; i >= 1; i --) infact[i - 1] = infact[i] * i % mod; // 逆推，时间复杂度能减少一半
}

ll C(int n, int m){
    if(n - m < 0 || n < 0 || m < 0) return 0;
    return fact[n] * infact[m] % mod * infact[n - m] % mod;
}

int main(){
    int n;
    cin >> n;
    init(2 * n);
    ll ans = (C(2 * n, n) + mod - C(2 * n, n - 1)) % mod;
    printf("%lld",ans);
    return 0;
}
```



#### 球盒模型

| $n$ 球 | $m$ 盒 | 是否允许为空 |             解              |
| :----: | :----: | :----------: | :-------------------------: |
|  相同  |  相同  |    不允许    |           划分数            |
|  相同  |  相同  |     允许     |           划分数            |
|  相同  |  不同  |    不允许    |   隔板法 $C_{n-1}^{m-1}$    |
|  相同  |  不同  |     允许     | 隔板法 $C_{n +m - 1}^{m-1}$ |
|  不同  |  相同  |    不允许    | 第二类斯特林数（容斥/递推） |
|  不同  |  相同  |     允许     | 第二类斯特林数（容斥/递推） |
|  不同  |  不同  |    不允许    |          容斥/递推          |
|  不同  |  不同  |     允许     |            $m^n$            |

1. **球相同，盒相同，不允许为空**

   将一个数 $n$ 分成 $m$ 个非负整数（划分数）递推求解。
   
   状态定义：$f[i][j]$ 用 $j$ 个盒子装 $i$ 个球的方案数。
   
   ​		转移方程：$f[i][j] = f[i][j - 1]$（新建一个盒子）$+ f[i-j][j]$（在每个盒子中放入一个球）。
   
   ​		答案：$f[n-m][m]$ 因为不能为空，就事先在每个盒子中放入一个球，还剩下 $n-m$ 个球自由放置的方案数。

```C++
// 将一个数 n 分成 m 个非负整数（划分数）递推求解
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 1e3 + 10, mod = 1e9 + 7;

ll n, m, f[N][N];
void solve(){
    f[0][0] = 1;
    for(int i = 1; i < N; i ++) f[0][i] = 1;
    for(int i = 1; i < N; i ++){
        for(int j = 1; j < N; j ++){
            if(i >= j) f[i][j] = (f[i - j][j] + f[i][j - 1]) % mod;
            else f[i][j] = f[i][j - 1];
        }
    }
}

int main(){
    solve();
    cin >> n >> m;
    if(n >= m) cout << f[n - m][m] << "\n";
    else cout << "0\n";
    return 0;
}
```



2. **球相同，盒相同，允许为空**

   ​		将一个数 $n$ 分成 $m，m-1,\dots,2,1$ 个非负整数（划分数）递推求解。按球相同，盒相同，不允许为空的递推公式，输出 $f[n][m]$ 即可。



3. **球相同，盒不同，不允许为空**

   ​		将 $n$ 个球排成一列，有 $m - 1$ 个隔板可以将球分成 $m$ 块，因为不能为空那么有 $n - 1$ 个空隙可以放入隔板，方案数为 $C_{n-1}^{m-1}$.



4. **球相同，盒不同，允许为空**

   ​		隔板法，球相同且盒子可以为空，所以板子可以相邻。考虑将球和板子看做同一个物品，共有 $n + m - 1$ 个空位，其中 $m-1$个位置放板子，剩下的位置放球，方案数为 $C_{n + m - 1}^{m-1}$.



5. **球不同，盒相同，不允许为空**

   解法一：递推求解

   状态定义：$F[i][j]$ 表示用 $j$ 个盒子来装前 $i$ 个球的方案数。
   
   ​		考虑第 $i$ 个球此时的状态，因为所有的盒子都是相同的，所以放入任意一个新的盒子方案数都是相同的，那么方案数为 $F[i - 1][j - 1]$ （表示前 $i - 1$个球放入 $j - 1$ 个盒子里，然后用第 $j$ 个盒子（其实用没装过小球的哪个盒子装都一样）来装第 $i$ 个小球）。
	
   ​		它也可以放入一个放过球的盒子里，因为球是不同的，所以放入 $j$ 个盒子的方案是不同的，即 $F[i - 1][j] * j$（表示前 $i - 1$ 个球放入 $j$ 个盒子里，然后第 $i$ 个小球放入 $j$ 个盒子中任意一个）。
	
   转移方程： $F[i][j] = F[i - 1][j - 1] + F[i - 1][j] * j$.
   
   因为不可以为空，所以最后的答案就是 $F[n][m]$.

```C++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 1e3 + 10, mod = 1e9 + 7;

ll n, m, f[N][N];

void solve(){
    f[0][1] = 1;
    for(int i = 1; i < N; i ++){
        for(int j = 1; j <= i; j ++){
            f[i][j] = (f[i - 1][j] * j % mod + f[i - 1][j - 1]) % mod;
        }
    }
}

int main(){
    solve();
    int n, m;
    cin >> n >> m;
    cout << f[n][m] << "\n";
    return 0;
}
```



   解法二：容斥

   求的是球不同，盒不同，不允许为空的解，最后除 $m!$ 即为所求。

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long

const int N = 1e6 + 10, mod = 1e9 + 7;

ll fact[N], infact[N];
ll C(int n, int m){
    return fact[n] * infact[m] % mod * infact[n - m] % mod;
}

ll ksm(ll a, ll b){ ll res; return res;}
int n, m;

int main(){
    cin >> n >> m;
    if(m > n) {
        cout << "0\n";
        return 0;
    }
    ll res = 0;
    for(int i = 0; i <= m; i ++){
        if(i % 2 == 0){
            res = (res + C(m, i) * ksm(m - i, n) % mod) % mod;
        }
        else{
            res = (res - C(m, i) * ksm(m - i, n) % mod + mod) % mod;
        }
    }
    res = res * infact[m] % mod;
    cout << res << "\n";
    return 0;
}
```



6. **球不同，盒相同，允许为空**

   ​		与上一个不同的是，盒子可以为空，而其他不变，因此 $F[i][j]$ 的状态不变，方程转移也不变，只有最后的答案变了。

   ​		因为盒子可以为空，所以最终的答案和盒子的数量无关，因此只需要枚举盒子数量，答案累加 $\sum_0^mF[n][i]$（即将 $n$ 个球放入 $i$ 个盒子）即可。

   

7. **球不同，盒不同，不允许为空**

   ​		考虑球不同，盒相同的情况，由于每个盒子不同了，那么任意两个盒子中的球互换就会形成新方案，答案就是 $m!*F[n][m]$.

   容斥解：即球不同，盒相同情况少除一个 $m!$.

   

8. **球不同，盒不同，允许为空**

   每个球都有 $m$ 种选择，答案为 $m^n$ ，快速幂求解。




### 多重集的组合数学

定义：多重集是指包含重复元素的广义集合。设 $S = \lbrace n_1\cdot a_1,n_2\cdot a_2,\ldots,n_k\cdot a_k \rbrace$ 表示由 $n_i$ 个 $a_i$ 组成的多重集。

#### 多重集的全排列

$S$ 的全排列数

设集合元素总数为 $n$，每次安排一种元素选位置，再让下一种元素在剩下的位置上选，依次进行。 

​								$ans = \prod_{i=1}^kC_{n-\sum_{j=1}^{i-1}n_j}^{n_i}=\frac{n!}{\prod_{i=1}^kn_i!}$

#### 多重集的组合数1

从多重集中选出 $r(r\leq n_i)$ 个元素的方案数

​		将问题转化为球盒模型，每种元素看做不同的盒子，选择元素相当于将相同的球放入不同的盒子。而每个盒子可以为空，且没有上限，这样问题就转化为：球同，盒不同，可以为空的方案数（具体证明见5.5.1.4球盒模型）。

​												$ans = C_{r+k-1}^{k-1}$

#### 多重集的组合数2/容斥

从多重集中选出 $r(r\geq n_i)$ 个元素的方案数

​		同样的将问题转化，每种元素看做不同的盒子，选择元素相当于将相同的球放入不同的盒子。此问题不同于一般的球盒模型，只有下界而无上界，即只需要考虑盒子是否为空不需要考虑最多能容纳多少个，此问题对于每个盒子有一个上界（每种元素数目有限）。

设每种元素选中的数目为 $r_i,0\leq r_i\leq n_i$.

解法一：容斥

若不考虑上界，方案等于多重集的组合数1，考虑减去不合法的方案。

设元素 $i$ 不满足的方案为 $s_i，r_i>n_i$ 

​								$ans=C_{r+k-1}^{k-1}-|s_1\cup s_2\cup\cdots\cup s_k|$ 

根据容斥原理将其展开

$ans=C_{r+k-1}^{k-1}-\sum_i^k|s_i|+\sum_{i<j}^k|s_i\cap s_j|-\cdots+(-1)^m\sum_{a_i<a_{i+1}}^k|\bigcap_{i=1}^ms_{a_i}|$ 

​		考虑如何求 $s_i$，$s_i$ 代表盒子中至少有 $n_i+1$ 个球方案，这恰好能用球盒模型解决（球盒模型只能解决有下界无上界的问题），将 $n$ 个球先分配 $n_i+1$ 个给第 $i$ 个盒子，将问题转化为将 $r-(n_i+1)$ 个球放入 $k$ 个盒子。

​												$s_i=C_{r-(n_i+1)+k-1}^{k-1}$

同理多个 $s_i$ 取交集方案为

​						$|s_i\cap s_j\cap\cdots\cap s_m|=C_{r-(n_i+1)-(n_j+1)\cdots-(n_m+1)+k-1}^{k-1}$

```C++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 21, mod = 1e9 + 7;

ll ksm(ll a, ll b){
    ll res = 1;
    while(b){
        if(b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

ll inv = 1;

ll C(ll n, ll m){
    if(n < 0 || m < 0 || m > n) return 0;
    ll res = 1;
    for(ll i = n; i >= n - m + 1; i --){
        res = res * (i % mod) % mod;
    }
    return res * inv % mod;
}

ll a[N];
int main(){
    ll n, m;
    cin >> n >> m;
    for(int i = 0; i < n; i ++) cin >> a[i];

    for(int i = 1; i < n; i ++) inv = inv * i % mod;
    inv = ksm(inv, mod - 2);
    
    ll ans = 0;
    for(int i = 0; i < (1 << n); i ++){
        ll sum = 0, p = 1; // sum:累加 (ai+1)，p:正负
        for(int j = 0; j < n; j ++){
            if(i >> j & 1){
                sum += (a[j] + 1);
                p *= -1;
            }
        }
        ll res = p * C(m - sum + n - 1, n - 1);
        ans = (ans + mod + res) % mod;
    }
    cout << ans << "\n";
    return 0;
}
```



#### 组合数的性质

（1）对选出的集合对全集取补集，故方案数不变（对称性）

​												$C_n^m=C_n^{n-m}$

（2）由定义导出递推式

​												$C_n^k=\frac nkC_{n-1}^{k-1}$

（3）组合数的递推式（杨辉三角的公式表达）

​												$C_n^m=C_{n-1}^m+C_{n-1}^{m-1}$

（4）二项式定理的特殊情况，取 $a=b=1$ 时成立

​												$\sum_{i=0}^nC_n^i=2^n$

（5）二项式定理的特殊情况，取 $a=1,b=-1$ 时成立

​												$\sum_{i=0}^n(-1)^iC_n^i=[n=0]$

（6）拆组合数

​										$\sum_{i=0}^mC_n^iC_m^{m-i}=C_{n+m}^m(n\geq m)$

（7）当（6）取 $n=m$ 时

​												$\sum_{i=0}^n(C_n^i)^2=C_{2n}^n$

（8）带权和，对（3）对应的多项式函数求导

​												$\sum_{i=0}^niC_n^i=n2^{n-1}$

（9）与上式同理

​												$\sum_{i=0}^ni^2C_n^i=n(n+1)2^{n-2}$

（10）对杨辉三角第 $k$ 列求和

​												$\sum_{i=0}^nC_k^i=C_{n+1}^{k+1}$

（11）通过组合意义上理解

​												$C_n^rC_r^k=C_n^kC_{n-k}^{r-k}$

（12）其中 $F$ 是斐波那契数列 

​												$\sum_{i=0}^nC_{n-i}^i=F_{n+1}$

（13）组合数的奇偶性判断：对于 $C_n^m$，若 $n$ &  $m=m$，则为奇数，否则为偶数

​												

## 整除分块

### 向下取整

求 $\sum_{i=1}^n \lfloor \frac ni\rfloor$.

对于每一个答案 $k = \lfloor \frac ni\rfloor$，会存在一个区间内所有的解都等于 $k$.

设使得上式成立的区间为 $[L,R]$，若已知 $L$ 则有：

​										$k=\lfloor \frac ni\rfloor=\lfloor \frac nL\rfloor$

​										$R = \lfloor \frac nk\rfloor=\lfloor \frac{n}{\lfloor\frac {n}{L}\rfloor}\rfloor$ 

例题：求 $\lfloor \frac ni\rfloor$ ($i\in[1,n]$) 中有多少不同的数，这些数中第 $k$ 大的是多少。

性质：当$i\leq\sqrt n$ 时除数肯定都不相同，剩下的数不同结果个数是 $\sqrt n$，除数个数为 $\frac n{\sqrt n + 1}$. 

```C++
// 求和⌊n/i⌋
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 2e5+5;
int main(){
    ll n,ans = 0;
    scanf("%lld",&n);
    for(ll l = 1, r; l <= n; l = r + 1){
        r = n / (n / l);
        ans += (n / l) * (r - l + 1);
    }
    printf("%lld",ans);
    return 0;
}
```



### 向上取整

对于每一个答案 $k = \lceil\frac ni\rceil$ 会存在一个区间内所有的解都等于 k.

设使得上式成立的区间为 $[L,R]$，若已知 $L$ 则有：

​												$R=\lfloor \frac{n-1}{\lfloor\frac{n-1}{L}\rfloor}\rfloor$

注意当除数为 $n$ 时要特判，不然会除 $0$ RE.



例题：给定范围 $[L,R]$，求任意两个数的 $\gcd$ 共有多少种。

考虑对于每种 $\gcd$ 存在的最小的一对数 $(x,y)$ 是什么

定理1：任意两个相邻的数 $\gcd=1$

定理2：$\gcd(x,y)=z\rightarrow \gcd(x*a,y*a)=z*a$

对于 $\gcd=a$，因为要保证不超出区间，找到最小的 $a$ 的倍数即

​												$x=\lceil\frac La\rceil*a$

不妨令 $y$ 为 $x+a$

​												$y=(\lceil\frac La\rceil+1)*a$

因为 $\gcd(\lceil\frac La\rceil,\lceil\frac L2\rceil+1)=1$，所以 $\gcd(x,y)=a$.

对于$\lceil\frac La\rceil$ 我们可以用整除分块确定一个除数相同的区间，再用二分确定除数相同的 $\gcd$ 的上界。

```C++
// https://codeforces.com/contest/1780/problem/E
#include <bits/stdc++.h>
using namespace std;
#define ll long long
ll L, R;
ll getnum(ll x){
    return (L / x + (L % x != 0) + 1) * x;
}
ll binary(ll l, ll r){
    while(l <= r){
        ll mid = (l + r) >> 1;
        if(getnum(mid) <= R) l = mid + 1;
        else r = mid - 1;
    }
    return r;
}
void solve(){
    cin >> L >> R;
    if(L == R){
        cout << "0\n";
        return ;
    }
    ll ans = max(0LL, R / 2 - L + 1);
    for(ll l = 1, r; l < L; l = r + 1){ // 枚举gcd
        r = (L - 1) / ((L - 1) / l); // [l,r] 除数相同的gcd区间
        if(getnum(l) <= R) ans += (binary(l, r) - l + 1); // 确定上界
    }
    
    cout << ans << "\n";
    return ;
}
```



## 线性代数

### 线性基

```C++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int MAXL = 62;

struct LinearBasis{
    ll a[MAXL + 5];
    LinearBasis(){
        fill(a, a + MAXL + 5, 0); // 初始化为全0
    }
};

void insert(ll t, ll a[]) { // 保证 j <= i中 只有 ai 的二进制第 i 位上为 1
    for (int i = MAXL; i >= 0; i --) {
        if (!(t & (1LL << i))) continue ;
        if (!a[i]) { a[i] = t; break; } // 选入线性基
        else t ^= a[i];
    }
}

bool ask(ll x, ll a[]) { //询问x是否能由该线性基构造出来
    for (int i = MAXL; i >= 0; i --) { 
        if (x & (1LL << i)) x ^= a[i]; // 根据插入的性质，若失败说明之前的基已经能把该数表达出来 即 x == 0
    }
    return x == 0;
}

ll get_min(ll a[]) { // 记得特判0
    for(int i = 0; i <= MAXL; i ++) {
        if(a[i]) return a[i]; // 之后的数最高位都大于当前，直接返回第一个有值的
    }
}

ll get_max(ll a[]) { // 求数集合中，任意数异或的最大值
    ll res = 0;
    for (int i = MAXL; i >= 0; i --) { 
        res = max(res, res ^ a[i]); // 因为只有 >= i 的 ai 的二进制第 i 位上可能为1， 而我们贪心的尽可能取当前res中没有最高位，可以写成取max的形式
    }
    return res;
}

void out(LinearBasis &B){ // 输出线性基的每一位
    for(int i = 0; i <= MAXL; i ++){
		vector<int>bit;
		for(int j = 0; j <= MAXL; j ++){
			bit.push_back(B.a[i] >> j & 1);
		}
		reverse(bit.begin(), bit.end());
		for(int b : bit) printf("%d",b);
		puts("");
	}
}

int n;
int main(){
    cin >> n;
    LinearBasis A;
    for (int i = 1; i <= n; i ++){
        ll x; cin >> x;
        insert(x, A.a);
    }

    cout << get_max(A.a);
    return 0;
}
```



## 杜教BM

给出**线性方程**的前几项（越多越好），可以推出第 $n$ 项。

```C++
#include <bits/stdc++.h>
using namespace std;
#define rep(i,a,n) for (int i=a;i<n;i++)
#define per(i,a,n) for (int i=n-1;i>=a;i--)
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
typedef vector<int> VI;
#define ll long long
const ll mod = 1000000007;
ll powmod(ll a,ll b)
{
    ll res = 1;
    a %= mod;
    assert(b >= 0);
    for(; b; b >>= 1)
    {
        if(b & 1) res = res * a % mod;
        a = a * a % mod;
    }
    return res;
}
int _,n;
namespace linear_seq {
    const int N = 10010;
    ll res[N],base[N],_c[N],_md[N];
    vector<int> Md;
    void mul(ll *a,ll *b,int k) {
        for(int i = 0; i < k + k; ++ i) _c[i] = 0;
        for(int i = 0; i < k; ++ i)
        {
            if(!a[i]) continue ;
            for(int j = 0; j < k; ++ j)
                _c[i+j] = (_c[i+j] + a[i] * b[j]) % mod; 
        }
        for(int i = k + k - 1; i >= k; i --)
        {
            if(!_c[i]) continue ;
            for(int j = 0; j < (int)Md.size(); ++ j)
                _c[i-k+Md[j]] = (_c[i-k+Md[j]] - _c[i] * _md[Md[j]]) % mod;
        }
        for(int i = 0; i < k; ++ i) a[i] = _c[i];
    }
    int solve(ll n,VI a,VI b){
        ll ans = 0,pnt = 0;
        int k = SZ(a);
        assert(SZ(a) == SZ(b));

        for(int i = 0; i < k; ++ i) _md[k-1-i] = -a[i]; 
        _md[k] = 1;
        Md.clear();

        for(int i = 0; i < k; ++ i)
            if (_md[i] != 0) Md.push_back(i);

        for(int i = 0; i < k; ++ i) res[i] = base[i] = 0;
        res[0] = 1;

        while ((1ll<<pnt) <= n)
        pnt ++;
        for(int p = pnt; p >= 0; p --) 
        {
            mul(res,res,k);
            if ((n>>p)&1) {
                for(int i = k - 1; i >= 0; i --) res[i+1] = res[i];
                res[0] = 0;
                for(int j = 0 ;j < (int)Md.size(); ++ j)
                    res[ Md[j] ]=(res[ Md[j] ] - res[k] * _md[Md[j]]) % mod;
            }
        }
        rep(i,0,k) ans = (ans + res[i] * b[i]) % mod;
        if (ans < 0) ans += mod;
        return ans;
    }
    VI BM(VI s) {
        VI C(1,1),B(1,1);
        int L = 0,m = 1,b = 1;
        for(int n = 0; n < (int)s.size(); ++ n) {
            ll d = 0;
            for(int i = 0; i < L + 1; ++ i)
                d = (d + (ll)C[i] * s[n-i]) % mod;
            if (d == 0) ++ m;
            else if (2*L<=n) 
            {
                VI T = C;
                ll c = mod - d * powmod(b, mod - 2) % mod;
                while (SZ(C) < SZ(B) + m) C.push_back(0);

                for(int i =0; i < (int)B.size(); ++ i)
                    C[i+m] = (C[i+m] + c * B[i]) % mod;
                L = n + 1 - L; B = T; b = d; m = 1;
            } 
            else 
            {
                ll c = mod-d*powmod(b, mod-2) % mod;
                while (SZ(C) < SZ(B) + m) C.push_back(0);

                for(int i = 0; i <(int) B.size(); ++ i)
                    C[i+m] = ( C[i+m] + c * B[i]) % mod;
                ++ m;
            }
        }
        return C;
    }
    ll gao(VI a,ll n) {
        VI c = BM(a);
        c.erase(c.begin());
        for(int i = 0 ; i < (int)c.size( ); ++ i)
            c[i] = (mod - c[i]) % mod;
        return (ll)solve(n,c,VI(a.begin(),a.begin() + SZ(c)));
    }
};
int main() {
    ll n;
    VI a;
    int N,v;
    a.push_back(3);
    a.push_back(9);
    a.push_back(20);
    a.push_back(46);
    a.push_back(106);
    a.push_back(244);
    a.push_back(560);
    a.push_back(1286);
    a.push_back(2956);
    a.push_back(6794);
    for (;~scanf("%lld",&n);)
    printf("%lld\n",linear_seq::gao(a,n - 1));
    return 0 ;
}
```





# 图论



## Tarjan

### 强连通分量有向图缩点

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;

vector<int>g[N], e[N]; //原图, 缩点后的新图 
int n, m, cnt, top, scc_cnt; //cnt:顺序时间戳，top：栈顶元素下标， inde:强连通分量的编号 
int low[N], dfn[N], Id[N], st[N]; //dfn:第一次到的时间 low:能访问到的最早的时间，Id编号  
bool vis[N];//标记元素是否在栈中 

void tarjan(int u){//求强连通分量 
	vis[u] = 1; st[top ++] = u;//标记入栈  
	dfn[u] = low[u] = ++ cnt;//时间戳 

	for(auto v : g[u]) {
		if(!dfn[v]){//新点未访问过 
			tarjan(v);
			low[u] = min(low[u] , low[v]);
		}
		else if(vis[v]) //新点访问过且此时在栈中 
			low[u] = min(low[u], dfn[v]); // (求割点割边时的错误写法：low[v])
	}

	if(dfn[u] == low[u]){//强连通分量的顶点
		scc_cnt ++;
		while(true){
			int x = st[-- top]; vis[x] = 0;
			Id[x] = scc_cnt;
			if(x == u)break; 
		}
	} 
}

void build(){//缩点后建新图 
	for(int u = 1; u <= n; u ++){
		for(auto v : g[u]){ //遍历原图 
			if(Id[u] != Id[v]){
				e[Id[u]].push_back(Id[v]); //如果两点Id编号不同说明不在同一个强连通分量中，建边 
			}
		}
	}
}

int main(){
	cin >> n >> m;
	for(int i = 1; i <= m; i ++){
		int u, v;
		cin >> u >> v;
		if(u == v) continue;
		g[u].push_back(v);
	}
	for(int i = 1; i <= n; i ++){
		if(!dfn[i]) tarjan(i);
	}
	//每个点都需要跑一遍，因为不能保证1号点可以连通所有点 
	build();
	return 0;
}
```



### 点双连通分量无向图缩点

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
struct Edge{
	int to, next;
}e[N * 2];

int head[N], tot = 1;
vector<int> bcc[N];
int bri[N], cut[N]; // 桥  割点
int n, m, bcc_cnt;
void add(int from, int to){
	e[++ tot] = {to, head[from]};
	head[from] = tot;
}

int vis[N], st[N], top;
int dfn[N], low[N], num;
void tarjan(int u, int in_edg){
	dfn[u] = low[u] = ++ num;
	vis[u] = 1; st[++ top] = u;
	int son = 0;
	for(int i = head[u]; i; i = e[i].next){
		int v = e[i].to;
		if(!dfn[v]){
			son ++;
			tarjan(v, i);
			low[u] = min(low[u], low[v]);
			if(dfn[u] < low[v])
				bri[i] = bri[i ^ 1] = 1;

			if(low[v] >= dfn[u]){
				cut[u] = 1;
				bcc_cnt ++;
				bcc[bcc_cnt].push_back(u);
				int x;
				do{
					x = st[top --];
					bcc[bcc_cnt].push_back(x);
				}while(x != v);
			}
		}
		else if(vis[v] && dfn[v] < dfn[u] && i != (in_edg ^ 1))
			low[u] = min(low[u], dfn[v]);
	}
	if(in_edg == 0 && son == 1) cut[u] = 0; // 特判根节点
}


int main(){
	ios::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	cin >> n >> m;
	tot = 1;
	for(int i = 1; i <= m; i ++){
		int u, v;
		cin >> u >> v;
		add(u, v);
		add(v, u);
	}

	for(int i = 1; i <= n; i ++){
		if(!dfn[i]) tarjan(i, 0);
    }

    cout << "\n";
	for(int i = 1; i <= bcc_cnt; i ++){
		for(auto v : bcc[i]) cout << v << " "; cout << "\n";
	}
	for(int i = 1; i <= n; i ++)
		if(cut[i]) cout << "cut " << i << "\n";
	return 0;
}
```



### 边双连通分量无向图缩点

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 1e5 + 10;
int n, m;

int dfn[N], low[N], bccno[N], bcc_cnt;

struct edge{
    int to, nex;
}e[N * 2];
int head[N], tot = 1, num;
void add(int to, int from){
    e[++ tot] = {to, head[from]};
    head[from] = tot;
}    

int st[N], vis[N], top;
void Tarjan(int u, int in_edg){ // 类似求割边
    dfn[u] = low[u] = ++ num;
    st[++ top] = u; vis[u] = 1;
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(!dfn[v]){
            Tarjan(v, i);
            low[u] = min(low[u], low[v]);
        }
        else if(i != (in_edg ^ 1) && vis[v]) // 防止重边 且在栈中
            low[u] = min(low[u], dfn[v]);
    }
    if(low[u] == dfn[u]){
        ++ bcc_cnt;
        while(true){
            int x = st[top --]; vis[x] = 0;
            bccno[x] = bcc_cnt;
            if(x == u) break;
        }
    }
}

vector<int> g[N];
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> m;
    for(int i = 1; i <= m; i ++){
        int u, v;
        cin >> u >> v;
        add(u, v); add(v, u);
    }

    Tarjan(1, 0);
    
    for(int u = 1; u <= n; u ++){
        for(int i = head[u]; i; i = e[i].nex){
            int v = e[i].to;
            if(bccno[u] != bccno[v]) g[bccno[u]].push_back(bccno[v]);
        }
    }
    return 0;
}
```



### 无向图求割点

​		割点判定法则：在无向图中，当某一点 $u$，其任意一个出边所连的点 $v$ 当且仅当满足 $dfn[u]\leq low[v]$（该点有一个子树中所有的点都没有办法追溯回 $u$ 的祖先，则当将 $u$ 删除时该子树会断开）时 $u$ 为该图的割点。特别的当 $u$ 为根节点时有两个及以上的不互相联通的子树时 $u$ 才为割点。

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e4 + 10,M = 1e5 + 10;
bool cur[N];//割点集
int n,m,dfn[N],low[N],tot,num,head[N];

struct edge{
    int to,nex;
}e[M * 2];
void add(int to,int from)
{
	e[++ tot].to = to;
	e[tot].nex = head[from];
	head[from] = tot; 
}
int root;
void tarjan(int u,int fa)
{
    dfn[u] = low[u] = ++ num;
    int son = 0;
    for(int i = head[u]; i; i = e[i].nex)
    {
        int v = e[i].to;
        if(!dfn[v]) 
        {
            tarjan(v , i);
            low[u] = min(low[u] , low[v]);
            if(low[v] >= dfn[u]) {
                son ++;
                if(u != root || son > 1) cur[u] = true;
            }
        }
        else if(v != fa)
            low[u] = min(low[u] , dfn[v]);//此处dfn[v]不能写成low[v]
    }
}

int main(){
    scanf("%d%d",&n,&m);
    for(int i = 1; i <= m; i ++){
        int u,v;
        scanf("%d%d",&u,&v);
        if(u == v) continue ;
        add(u, v), add(v, u);
    }
    for(int i = 1; i <= n; i ++)
        if(!dfn[i]) root = i, tarjan(i , 0);
    
    int ans = 0;
    for(int i = 1; i <= n; i ++)
        if(cur[i]) ans ++;
    printf("%d\n",ans);
    for(int i = 1; i <= n; i ++) 
        if(cur[i]) printf("%d ",i);
    return 0;
}
```



### 无向图求割边

​		割边判定法则：在无向图中，与求解割点类似的在深度优先搜索生成树中某一条边 $(u,v)$，当且仅当 $dfn[u]<low[v]$ 时，该边为割边。

注意图中有重边时要用是否为同一条边的反向边来判断是否更新 $dfn$.

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;

int dfn[N];//当前节点的访问时间戳（或者说是访问顺序）一经赋值就不再改变 
int low[N];//当前时间点能 连通到的 除父节点外 的点的最早时间戳 
struct edge{
	int to,nex;
}e[N * 2];
bool bridge[N];//记录割边

int n,m,tot = 1,num,head[N];
void add(int to,int from)
{
	e[++ tot] = {to, head[from]}
	head[from] = tot; 
}

void tarjan(int u, int in_edg)//u当前节点 ，fa父节点用于判断割点时使用 
{
	dfn[u] = low[u] = ++ num;
	for(int i = head[u]; i; i = e[i].nex)
	{                                                                                           
		int v = e[i].to;
		if(!dfn[v])
        {
			tarjan(v , i);
			low[u] = min(low[u], low[v]); 
			if(low[v] > dfn[u]) 
				bridge[i] = bridge[i ^ 1] = true;
		}
		else if(i != (in_edg ^ 1))//当不是同一条边时才更新(v != fa不能解决重边的问题)
			low[u] = min(low[u] , dfn[v]);
	}
}

int main(){
	tot = 1;
	scanf("%d%d",&n,&m);
	for(int i=0;i<m;i++){
		int u,v;
		scanf("%d%d",&u,&v);
		add(u, v);
		add(v, u);
	}
	for(int i = 1; i <= n; i ++)
		if(!dfn[i]) tarjan(i, 0);
	
	int ans = 0;
	for(int i = 2; i <= tot; i += 2)
		if(bridge[i]) ans ++;
	printf("%d",ans);
	return 0;
}
```



### 2-SAT

```C++
#include <iostream>
using namespace std;
int const N = 1e6 + 10;
struct Edge{
    int to,nex;
}e[N * 2];
bool vis[N * 2];
int Id[N * 2],head[N * 2],tot,id,n,m;
int st[N * 2], low[N * 2], dfn[N * 2];
void add(int from,int to)
{
    e[++tot].to = to;
    e[tot].nex = head[from];
    head[from] = tot; 
}
void init(int u,int F1,int v,int F2)
{
    if(!F1&&!F2)//都假
    {
        add(u,v+n);     //a=1则b=0 
        add(v,u+n);     //b=1则a=0
    }
    else if(!F1&&F2)//u假v真
    {
        add(u,v);       //a=1则b=1
        add(v+n,u+n);   //b=0则a=0 
    }
    else if(F1&&!F2)//u真v假
    {
        add(v,u);       //a=0则b=0
        add(u+n,v+n);   //b=1则a=1
    }
    else if(F1&&F2)//都真
    {
        add(u+n,v);     //a=0则b=1
        add(v+n,u);     //b=0则a=1 
    }
}
int cnt,top;
void tarjan(int u)//求出强连通分量的编号，即代表反的拓扑序
{
    if(dfn[u]) return ;
    vis[u] = 1, st[top++] = u;
    dfn[u] = low[u] = ++cnt;
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(!dfn[v]){
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else if(vis[v])
            low[u] = min(low[u], dfn[v]);
    }
    if(dfn[u] == low[u]){
        id ++;
        while(true){
            int x = st[--top]; vis[x] = 0;
            Id[x] = id; 
            if(x == u)break;
        }
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i = 0; i < m; i ++){
        int a,b,x,y;
        scanf("%d%d%d%d",&a,&x,&b,&y);
        init(a,x,b,y);
    }
    for(int i = 1; i <= n*2; i ++) tarjan(i);//对每个点求，可能是非连通图
    for(int i = 1; i <= n; i ++){
        if(Id[i] == Id[i+n] && Id[i] != 0){
            printf("IMPOSSIBLE");
            return 0;
        }
    }
    printf("POSSIBLE\n");
    for(int i = 1; i <= n; i ++)//在逆拓扑序中强连通分量编号越小，拓扑序越大，越优
    {
        if(Id[i] < Id[i+n])printf("1 ");
        else printf("0 ");
    }
    return 0;
}
```



## Boruvka最小生成树

​		Boruvka 最小生成树算法一般适用于边权由点权决定的最小生成树，可以看做是一个多路增广的Kruskal，一般流程为：

初始的 $n$ 个点看做 $n$ 个连通块

1. 对于每一个连通块，找到一条由它内部连向其他连通块的最小边

2. 将这些边连通即加入最小生成树中去

3. 所有点连通则退出

​        难点一般在于1. 如何维护每个连通块的最小边，不同的题维护的方法也不同，本题也没有固定的模板，只能参考思路。

​		根据启发式合并的思想，每次连边后连通块的数量会减半，所以最多只需要做 $logn$ 次。设每次找到连通块的最小边时间为 $O(m)$，合并复杂度为 $O(n)$，总时间复杂度为 $O((n + m)logn)$ .



例题：最小异或生成树

​		给定 $n$ 个点的完全图，点 $i$ 的权值为 $a_i$，点 $i$ 与 $j$ 之间边的权值为 $a_i\bigoplus a_j$ ，求最小生成树边权之和。

​		考虑将所有点权 $a_i$ 插入到 Trie 树中， 要使得异或值尽可能小就需要从高到低尽量走同一条路径。

​        树形结构从叶子节点向上合并的过程就可以很形象的展示这一思路，每次找边权最小的两个点使连通块合并的情况。因为优先合并从高到低相同的，一个节点的左右儿子就是两个不同的连通块，我们要将他们合并。		具体如何合并我们可以采用启发式合并的思想。对于一个有两个儿子的节点，需要找到两个儿子中异或最小的两个节点连边，于是我们枚举 $siz$ 较小的一边，拿去 $siz$ 较大的一边查询合并。

​		时间复杂度：每个节点最多会被查询 $logn$ 次，而字典树每次查询的时间复杂度是 $logn$，总时间复杂度为 $O(n{logn}^{2})$.

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 2e5 + 10, inf = (1 << 30);

int son[N * 30][2], siz[N * 30], tot;

void insert(int x){
    for(int i = 30, p = 0; i >= 0; i --){
        int u = x >> i & 1;
        if(!son[p][u]) son[p][u] = ++ tot;
        p = son[p][u];
        siz[p] ++;
    }
}

int get_min(int x, int s, int p){
    int ans = 0;
    for(int i = s; i >= 0; i --){
        int u = x >> i & 1;
        if(son[p][u]) p = son[p][u];
        else {
            p = son[p][!u];
            ans += (1 << i);
        }
    }
    return ans;
}

ll ans;
int query(int p1, int x, int i, int p2, int s){ // 枚举siz较小的那边的数
    int res = inf;
    if(son[p1][0]) res = min(res, query(son[p1][0], x, i - 1, p2, s));
    if(son[p1][1]) res = min(res, query(son[p1][1], x + (1 << i - 1), i - 1, p2, s));
    if(!son[p1][0] && !son[p1][1]) res = get_min(x, s, p2); // 在siz较大的那边进行查询
    return res; 
}

void dfs(int p, int i){
    if(son[p][0]) dfs(son[p][0], i - 1);
    if(son[p][1]) dfs(son[p][1], i - 1);
    if(son[p][0] && son[p][1]) { // 需要将两个儿子合并
        if(siz[son[p][0]] < siz[son[p][1]]) ans += query(son[p][0], 0, i - 1, son[p][1], i - 2) + (1 << i - 1);
        else ans += query(son[p][1], 1 << (i - 1), i - 1, son[p][0], i - 2) + (1 << i - 1);
    }
}
int main(){
    int n;
    cin >> n;
    for(int i = 1, x; i <= n; i ++){C++
        cin >> x; insert(x);
    }
    dfs(0, 31);
    cout << ans;
    return 0;
}
```



## Kruskal重构树

kruskal 重构树性质

1. 它是一个二叉堆。
2. 若边权升序，则它是一个大根堆。
3. 任意两点路径边权最大值为Kruskal重构树上LCA的点权。

例题1：给定一个无向连通图，每个点和每条边都有一个权值，经过一个点可以获得该权值（不可重复获得），经过一条权值为 $w_i$ 的边需要当前手中的权值 $\geq w_i$（手中权值不会减少）。$q$ 个询问，每次给定初始出发点和初始权值，问最后最多能获得多少权值

```C++
// 题目：https://ac.nowcoder.com/acm/contest/24872/H 46th ICPC 上海 H
/*
思路：利用Kruskal重构树的性质， 升序边重构大根堆
从当前点向上走一次到达一个祖先节点， 因为是大根堆说明该棵子树中所有边都小于该祖先节点，于是就能获得该子树中所有权值
每次倍增向上跳，跳到最终能到的点树上前缀 + k 即为所求
*/

#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> pii;
const int N = 2e5 + 10, M = 1e5 + 10; // N 为点数 * 2 M 为边数

struct DSU {
    int p[N];
     
    DSU() {}
    DSU(int n) {
        init(n);
    }
     
    void init(int n) {
        for(int i = 1; i <= n; i ++) p[i] = i;
    }
     
    int find(int x) {
        while (x != p[x]) {
            x = p[x] = p[p[x]];
        }
        return x;
    }
     
    bool same(int x, int y) {
        return find(x) == find(y);
    }
     
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        p[y] = x;
        return true;
    }
};

struct edge{
    int u, v, w;
    bool operator < (const edge &A)const { return w < A.w; }
}e[M];

int n, m, cnt, val[N], pat[N], a[N];
pii ch[N]; // 二叉树存两个儿子节点

void kruskal(){ //两点之间的连边变作父亲节点
    DSU dsu(2 * n);
    sort(e + 1, e + m + 1);
    for(int i = 1; i <= cnt; i ++) ch[i] = {0, 0};

    cnt = n;
    for(int i = 1; i <= m; i ++){
        int u = dsu.find(e[i].u);
        int v = dsu.find(e[i].v);
        if(u == v) continue ;
        val[++ cnt] = e[i].w;
        dsu.merge(cnt, u);
        dsu.merge(cnt, v);
        pat[u] = pat[v] = cnt;
        ch[cnt] = {u, v};
    }
}

int pre[N], f[N][21], st[N][21]; // 维护前缀和  lca  val[pat[u]] 与 pre[u]的最大差值
void dfs(int u){
    if(!u) return ;

    auto [ls, rs] = ch[u];
    dfs(ls); dfs(rs);
    pre[u] += pre[ls] + pre[rs];

    f[u][0] = pat[u];
    st[u][0] = val[pat[u]] - pre[u];
}

void lca(){ // 先获得f[u][0] st[u][0] 再预处理出倍增
    for(int i = 1; i < 20; i ++){
        for(int j = 1; j < cnt; j ++){
            f[j][i] = f[f[j][i - 1]][i - 1];
            st[j][i] = max(st[j][i - 1], st[f[j][i - 1]][i - 1]);
        }
    }
}

int get_ans(int u, int k){
    for(int i = 19; i >= 0; i --){
        if(f[u][i] && st[u][i] <= k) u = f[u][i];   
    }
    return pre[u] + k;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int q;
    cin >> n >> m >> q;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        pre[i] = a[i];
    }
    for(int i = 1; i <= m; i ++){
        auto &[u, v, w] = e[i];
        cin >> u >> v >> w;
    }

    kruskal();
    dfs(cnt);
    lca();
    // for(int i = n + 1; i <= n * 2; i ++){ 当在不一定连通的图上重构时，重构出的可能是森林
    //     if(!siz[i]) dfs(i); 对于未遍历过的点需要处理
    // }
    for(int i = 1; i <= q; i ++){
        int u, k;
        cin >> u >> k;
        cout << get_ans(u, k) << "\n";
    }
    return 0;
}
```



例题2：[A-Tree_2023牛客暑期多校训练营6 ](https://ac.nowcoder.com/acm/contest/57360/A)树上任意黑白点对的价值为其简单路径上的最大边权， 可以花费 $cost_i$ 改变点 $i$ 的颜色，询问任意黑白点对价值之和最大为多少。

思路：利用重构树的性质3 重构后每个重构点的点权都是分别在两个儿子子树中点对的权值， 所以跑树形背包即可。



## 匹配问题

### 二分图匹配

匈牙利算法：时间复杂度 $O(V*E)$ 其中 $V$ 为点集，$E$ 为边集。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 510, M = 1e5 + 10;

struct Edge{
    int to, nex;
}e[M];

int n1,n2,m,head[N],tot;
void add(int to,int from){
    e[++ tot] = {to, head[from]};
    head[from] = tot;
}

int x[N], st[N]; //x[i]：代表左半边点i匹配到的右半边点编号
void init(){
    for(int i = 1; i <= n2; i ++) st[i] = 0;
}
bool dfs(int u){
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(st[v]) continue;//防止重复搜索死循环
        st[v] = 1;
        if(!x[v] || dfs(x[v])){//如果没有匹配，或者之前匹配的能更换
            x[v] = u;
            return true;
        }
    }
    return false;
}

int main(){
    scanf("%d%d%d",&n1,&n2,&m);
    for(int i = 1; i <= m; i ++){
        int u, v;
        scanf("%d%d",&u,&v);
        add(v,u);//无向图，但我们也只需要从一边的点集去匹配就行了
    }
    int ans = 0;
    for(int i=1;i<=n1;i++){
        init();
        if(dfs(i)) ans ++;
    }
    printf("%d",ans);
    return 0;
}
```



### 转化为网络最大流模型

​		假设转化后的图有 $n$ 个顶点，$m$ 条边，时间复杂度为 $O(\sqrt{n}m)$. 建图时注意，所有的边都需要建立为双向边，正流向为 $1$，逆流向 $0$.

​		因为建图时可能是由 点集 $V_1$ 向点集 $V_2$ ，两边点集可能都是从 $1$ 开始编号，所以需要将其中一个离散到 $|V_1| + id$ ，所以开空间容量和清空时需要谨慎。



### 最小路径覆盖问题

定义：在有向图中，找出最少的路径条数，使得这些路径经过了所有的点。

问题可以分为两类

1. 最小不相交路径覆盖：每一条路径经过的顶点各不相同。

2. 最小可相交路径覆盖：每一条路径经过的顶点可以相同。

特别的，每个点自己也可以称为是路径覆盖，只不过路径的长度是 $0$.

例如有向图：

![](C:\Users\饕餮\Desktop\肖天赐的ACM模板\图片\最小路径覆盖.png)

对于问题一：选择一条路径 $1\rightarrow3\rightarrow4$ 以后因为 $3$ 已经选过了，剩下的路径只能分别选择 $2$，$5$， 最少路径数为 $3$.

对于问题二：就可以选择 $1\rightarrow3\rightarrow4$ 和 $2\rightarrow3\rightarrow5$ 这样两条,最少路径数为 $2$.



问题一算法：

​		把原图的每个点 $V$ 拆成 $Vx$ 和 $Vy$ 两个点，如果有一条有向边 $A\rightarrow B$，那么就加边 $Ax\rightarrow By$. 这样就得到了一个二分图。**最小路径覆盖 = 原图的结点数 - 新图的最大匹配数。**

证明：

​		一开始每个点都是独立的为 $1$ 条路径，总共有 $n$ 条不相交路径。我们每次在二分图里找 $1$ 条匹配边就相当于把两条路径合成了 $1$ 条路径，也就相当于路径数减少了 $1$. 所以找到了几条匹配边，路径数就减少了多少。



问题二算法：

​		先求出原图的传递闭包，即如果 $a$ 到 $b$ 有路径，就加边 $a\rightarrow b$，然后就转化成了最小不相交路径覆盖问题。



## 网络流

### Dinic算法

​		时间复杂度为：$O(|E|*min(|E|^{1/2}, |V|^{2/3}))$ ，其中 $E$ 为边集， $V$ 为点集。 在单位容量网络上的时间复杂度参考二分图最大匹配最大流的时间复杂度。

```c++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const ll inf = 1e18;
const int N = 210, M = 5010;

int n,m,s,t,head[N],tot = 1; //注意要从1开始 即第一条边存储在e[2]中
struct edge{
	int to, nex, w;
}e[M * 2];
void add(int from,int to,int w){
	e[++ tot] = {to, head[from], w};
	head[from] = tot;
}

int dep[N];//用bfs分层
bool bfs()//判断是否还存在增广路
{
	queue<int>q;
	q.push(s);
	for(int i = 1; i <= n; i ++) dep[i] = 0;
	dep[s] = 1;
	while(!q.empty()){
		int u = q.front(); q.pop();
		for(int i = head[u]; i; i = e[i].nex){
			int v = e[i].to;
			if(e[i].w && !dep[v]){
				dep[v] = dep[u] + 1;
				if(v == t) return true;
				q.push(v);
			}
		}
	}
	return dep[t];
}

ll dfs(int u, ll inflow)//in为进入的流,源点的流无限大
{
	if(u == t)//到达汇点
		return inflow;//返回这一条增广路的流量
	ll outflow = 0;
	for(int i = head[u]; i && inflow; i = e[i].nex)//还有残余流量
	{
		int v = e[i].to;
		if(e[i].w && dep[v] == dep[u] + 1){//只搜索下一层次的点，防止绕回或走反向边
			ll flow = dfs(v , min(1ll * e[i].w,inflow));//min选取边残余容量与入流残余的最小值
			e[i].w -= flow; //减去达到汇点的增广流量
			e[i ^ 1].w += flow; //反向边增加流量
			inflow -= flow; //入流减少相同的
			outflow += flow; //增广路增加流量
		}
	}
	if(outflow == 0) dep[u] = 0;//通过u点不能到达汇点，剪枝
	return outflow;
}

int main(){
	scanf("%d%d%d%d",&n,&m,&s,&t);
	for(int i = 0; i < m; i ++)
	{
		int u,v,w;
		scanf("%d%d%d",&u,&v,&w);
		add(u,v,w);
		add(v,u,0);
	}
	ll ans = 0;
	while(bfs()) 
        ans += dfs(s ,inf);
	printf("%lld",ans);
	return 0;
}
```





## 树哈希

### 方法一

​		树hash的一般形式为 $h(x) = \sum f(h[s]) % mod$ $s$ 为 $x$ 的子节点。其中 $h(x)$ 为以 $x$ 为根的子树的 hash 值， $f(x)$ 为多重集的hash函数，为整数到整数的映射。若要换根，第二次dp时将子树hash减去即可。

```C++
// 第2种计算hash的方式：将子树的hash值排序后，按字符串hash的方式顺序hash
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long
const int N = 1e6 + 10;

const ull mask = std::chrono::steady_clock::now().time_since_epoch().count();
ull shift(ull x) { //自然溢出取模
  	x ^= mask;
  	x ^= x << 13;
  	x ^= x >> 7;
  	x ^= x << 17;
  	x ^= mask; //预防卡 xor hash 异或随机常数
  	return x;
}

ull Hash[N];
vector<int>g[N];
void getHash(int u, int fa){
    Hash[u] = 1;
    for(int v : g[u]){
        if(v == fa) continue ;
        getHash(v, u);
        Hash[u] += shift(Hash[v]);
    }
}

int n;
int main(){
    cin >> n;
    for(int i = 1; i < n; i ++){
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    getHash(1, 0);
    return 0;
}
```



## 曼哈顿距离与切比雪夫距离

**定义：**设在二维平面坐标系中两点 $p1 = (x1, y1)$，$p2 = (x2, y2)$。

曼哈顿距离：$dis = \vert x1 - x2\vert + \vert y1 - y2\vert$。

切比雪夫距离：$dis = \max(\vert x1 - x2\vert, \vert y1 - y2\vert)$。



### 曼哈顿距离与切比雪夫距离的相互转换

对于坐标系中的所有点集 $p(x, y)$，

将点 $p(x,y)$ 的坐标变为 $p'(x+y,x−y)$ 后，原坐标系中的曼哈顿距离 $=$ 新坐标系中的切比雪夫距离。

将点 $p(x,y)$ 的坐标变为 $p’(\frac{x+y}{2},\frac{x-y}{2})$ 后，原坐标系中的切比雪夫距离 $=$ 新坐标系中的曼哈顿距离。



### 任意两点的 曼哈顿/切比雪夫 距离之和

​		因为只需要求任意两点曼哈顿距离之和，所以直接将 $x, y$ 坐标分别排序，使用前缀和单个点的曼哈顿距离之和为 $O(1)$，总复杂度 $O(n)$.

​		而求切比雪夫距离单个点 $O(n)$，总复杂度 $O(n^2)$. 于是我们经常转化为曼哈顿距离来求，同时为了避免精度问题，转化时不除 $2$，而是将最后的答案除 $2$.

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 1e6 + 10;
int n, x[N], y[N], sx[N], sy[N];
ll pre[N];

ll get_mat(int p[]){//其中一维的曼哈顿距离
    sort(p + 1, p + 1 + n);
    for(int i = 1; i <= n; i ++){
        pre[i] = pre[i - 1] + p[i];
    }
    ll dis = 0;
    for(int i = 1; i <= n; i ++){
        dis += 1ll * p[i] * (i - 1) - pre[i - 1];
        dis += pre[n] - pre[i] - 1ll * p[i] * (n - i);
    }
    return dis;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    cin >> n;
    for(int i = 1; i <= n; i ++) cin >> x[i];
    for(int i = 1; i <= n; i ++) cin >> y[i];
    for(int i = 1; i <= n; i ++){ //将切比雪夫转化为曼哈顿
        sx[i] = x[i] + y[i]; 
        sy[i] = x[i] - y[i];
    }
    
    ll mat = get_mat(x) + get_mat(y);
    ll cbs = get_mat(sx) / 2 + get_mat(sy) / 2; //得到答案时再除2
    cout << mat - cbs;
    return 0;
}
```



## 欧拉图的判定与相关定理

定义：图 $G$ 中经过所有边**恰好一次**的路径叫**欧拉路径**，如果此路径的**起点**和**终点**相同，则称其为一条**欧拉回路**。

**欧拉图：**有欧拉回路的图。

**半欧拉图：**有欧拉路径但无欧拉回路的图。

欧拉图及半欧拉图的判定：

定理1：无向图 $G$ 是欧拉图，当且仅当 $G$ 的非零度顶点连通，且所有顶点度数都是偶数。

定理2：有向图 $G$ 是欧拉图， 当且仅当 $G$ 的**非零度顶点是强连通的**，且所有顶点出度等于入度。

定理3：无向图 $G$ 是半欧拉图，当且仅当 $G$ 的非零度顶点连通，且恰有两个顶点度数为奇数（两者一定是欧拉路径的起点和终点）。

定理4：有向图 $G$ 是半欧拉图，，当且仅当 $G$ 的**非零度顶点是弱连通的**，且前有两个顶点入度不等于出度，其中一点出度比入度大$1$，为路径起点，另一点入度比出度大 $1$，为路径的终点。

### 欧拉路径

例题：判断图 $G$ 是否有欧拉路径，并找到字典序最小的欧拉路径。

​		欧拉图也存在欧拉路径，先将非欧拉图且非半欧拉图的判否。再贪心的用邻接表存图并排序，保证每一步都是当前最序号最小的点以此满足字典序最小。

特例：一条链 + 链上一个环

![](C:\Users\饕餮\Desktop\肖天赐的ACM模板\图片\欧拉路径.jpg)

​		若先走编号小的点（$1 \rightarrow 2 \rightarrow 3\rightarrow 4$），就不能回头走环了（ $2\rightarrow3\rightarrow5\rightarrow2$）于是我们在结束搜索时才记录节点，最后倒序输出，最后就算错过环先走链，回溯时也会再走一次环，此时因为环后遍历到，处在答案序列的后端，倒序以后就形成先走环的局面。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;

struct DSU {
    std::vector<int> f, siz;
     
    DSU() {}
    DSU(int n) {
        init(n + 1);
    }
     
    void init(int n) {
        f.resize(n); // 重构容器大小到n
        std::iota(f.begin(), f.end(), 0); // 批量递增赋值
        siz.assign(n, 1); // 赋值n个1
    }
     
    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
     
    bool same(int x, int y) {
        return find(x) == find(y);
    }
     
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
};


int n, m, s = 1, t;
int vis[N * 2], in[N], out[N];
deque<int> g[N];

vector<int> ans;
void dfs(int u){
    while(!g[u].empty()){
	    int v = g[u].front();
	    g[u].pop_front(); // 每条边只走一次 pop掉
	    dfs(v);
    }
    ans.push_back(u); // 结束搜索才记录进答案序列
}

int main(){
    cin >> n >> m;
    DSU dsu(n + 1);

    int cnt = n;
    for(int i = 1; i <= m; i ++){
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        in[v] ++, out[u] ++;
        if(dsu.merge(u, v)) cnt --; 
    }
	
    for(int i = 1; i <= n; i ++){ // 减去0度顶点
        if(!in[i] && !out[i]) cnt --;
    }

    if(cnt != 1){ // 判连通
        cout << "No\n";
        return 0;
    }

    int sum = 0;
    for(int i = 1; i <= n; i ++){ // 判起点终点是否唯一
        if(in[i] == out[i]) continue ;
        if(out[i] - in[i] == 1) s = i;
        else if(in[i] - out[i] == 1) t = i;
        sum ++;
    }
    if(sum > 2 || (sum && (!s || !t))){
        cout << "No\n";
        return 0;
    }

    for(int i = 1; i <= n; i ++){
        sort(g[i].begin(), g[i].end()); // 贪心排序
    }
    
    dfs(s);
    reverse(ans.begin(), ans.end());
    for(auto v : ans) cout << v << " "; // 倒序输出
    return 0;    
}
```



## 哈密顿图的判定与相关定理

定义：图 $G$ 中经过所有顶点**恰好一次**的路径叫**哈密顿通路**，如果此路径的**起点**和**终点**相同，则称其为一条**哈密顿回路**。

**哈密顿图：**有哈密顿回路的图。

**半哈密顿图：**有哈密顿通路但无哈密顿回路的图。

哈密顿图和半哈密顿图的判定：

定理1：设 $G$ 是大小为 $n(n\geq2)$ 的无向简单图，若对于 $G$ 中任意不相邻的顶点 $u, v$，均有 $d(u) + d(v) \geq n-1$，则 $G$ 存在哈密顿通路。

定理2：设 $G$ 是大小为 $n(n\geq3)$ 的无向简单图，若对于 $G$ 中任意不相邻的顶点 $u, v$，均有 $d(u) + d(v) \geq n$，则 $G$ 存在哈密顿回路。

定理3：设 $G$ 是大小为 $n(n\geq3)$ 的无向简单图，若对于 $G$ 中任意顶点 $u$，均有 $d(u)\geq \frac n2$，则 $G$ 存在哈密顿回路。

竞赛图定义：边数为 $\frac {n*(n-1)}{2}$ 的无重边自环的有向图，也可以认为是给无向完全图的每条边定向后的图

定理4：设 $G$ 是竞赛图，则 $G$ 具有哈密顿通路，强连通的竞赛图为具有哈密顿回路。



# 数据结构



## 单调栈模板

```C++
//求 i 个元素之后第一个大于 a[i] 的元素的下标 r[i] 为所求
int cnt = 0;
for(int i = 1; i <= n; i ++){
    while(cnt && a[st[cnt]] < a[i]) r[st[cnt --]] = i;
    st[++ cnt] = i;
}
```



## 单调队列模板

```C++
//队内存的数单调递增，队首是最小值，维护区间最小值。 反之则是区间最大值
int tot = 0, top = 1;
for(int i = 1; i <= n; i ++){
    while(tot >= top && a[q[tot]] > a[i])tot --;//将更劣的解从队列弹出
    q[++ tot] = i;//存入本次解
    while(i - q[top] + 1 > k) top ++;//将队首不合法的解弹出
    if(i >= k)printf("%d ",a[q[top]]);
}
```



## 倍增

### ST表（倍增预处理求静态区间最值）

```C++
/*
f[i][j]:代表从i位置开始的2^j个元素的最值
f[i][j] = max(f[i][j-1],f[i+(1<<j-1)][j-1])，即将[i,i+(2^j)-1]区间一分为二取最值

查询的时候对于区间[l,r] 我们可以选择一个幂次k l+(2^k)-1<=r
选择[f[l][k],f[r-(1<<k)+1][k]] 即可将 [l,r] 区间覆盖虽然有重叠但不会有遗漏
*/
#include<iostream>
#include<algorithm>
using namespace std;
const int N=1e5+5;
int f1[N][20],f2[N][20];//分别维护最大最小值
int a[N],lo[N];
void ST(int n){
	for(int i = 2; i <= n; i ++) lo[i] = lo[i>>1] + 1;//预处理出log2数值，可减低一半复杂度

	for(int j = 1; (1<<j) <= n; j ++){
		for(int i = 1; i + (1<<j) - 1 <= n; i ++){
			//2^j=2^(j-1)+2^(j-1)//前后各一半
			f1[i][j] = min(f1[i][j-1], f1[i+(1<<j-1)][j-1]);
			f2[i][j] = max(f2[i][j-1], f2[i+(1<<j-1)][j-1]);
		}
	}
}

int get_max(int l, int r){
	int k = lo[r-l+1];
	return max(f2[l][k], f2[r-(1<<k)+1][k]);
}
int get_min(int l, int r){
	int k = lo[r-l+1];
	return min(f1[l][k], f1[r-(1<<k)+1][k]);
}
int quer_x(int l,int r){
	int k = lo[r-l+1]; //int k=log2(r-l+1);未预处理的求法，多次询问大量重复计算
	//用两个最大的不超过查询区间的二进制倍数区间覆盖
	return max(f2[l][k], f2[r-(1<<k)+1][k]) - min(f1[l][k], f1[r-(1<<k)+1][k]);
}

int main(){
	int n,m,k;
	scanf("%d%d",&n,&m);
	for(int i = 1; i <= n; i ++){
		scanf("%d",&a[i]);
		f1[i][0] = f2[i][0] = a[i];
	}
	ST(n);//预处理
	for(int i = 0; i < m; i ++){
		int l,r;
		scanf("%d%d",&l,&r);
		printf("%d\n",quer_x(l,r));
	}
	return 0;
}
```



### 倍增求LCA

```C++
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 5e5 + 10;
vector<int>g[N];
int dep[N],f[N][25];
void dfs(int u,int fa)
{
    f[u][0] = fa; 
    dep[u] = dep[fa] + 1;
    for(int i = 1; (1 << i) <= dep[u]; i ++)
        f[u][i] = f[f[u][i - 1]][i - 1];

    for(int v : g[u])
        if(v != fa) dfs(v , u); 
}

int LCA(int u, int v)
{
    if(dep[u] < dep[v]) swap(u , v);
    int d = dep[u] - dep[v]; // 记录深度差
    for(int i = 0; i <= 20; i ++){
        if((1 << i) & d) u = f[u][i];// 如果深度差包含2^i 那么就向上跳2^i
    }
    if(u == v) return u; // 如果相同说明lca是u

    /*当跳到的点相同时不能认为f[u][i]就是lca,可能只是公共祖先而不是最近公共祖先,而通过二进制组合一定能跳到最近公共祖先的儿子一层，此时f[u][0]为答案*/
    for(int i = 20; i >= 0; i --) // 同一深度同时向上跳
    {
        if(f[u][i] != f[v][i])
        {
            u = f[u][i];
            v = f[v][i];
        }
    }
    return f[u][0];
}
int main()
{
    int n,m,root;
    scanf("%d%d%d",&n,&m,&root);
    for(int i = 1; i < n; i ++)
    {
        int u,v;
        scanf("%d%d",&u,&v);
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs(root , 0);
    for(int i = 1; i <= m ; i ++)
    {
        int u,v;
        scanf("%d%d",&u,&v);
        printf("%d\n",LCA(u , v));
    }
    return 0;
}
```



## 并查集

### 结构体封装模板

```C++
#include <bits/stdc++.h>
using namespace std;
struct DSU {
    std::vector<int> f, siz;
    
    DSU() {}
    DSU(int maxn) {
        init(mn);
    }
     
    void init(int maxn) {
        f.resize(++ maxn); // 重构容器大小到 n
        std::iota(f.begin(), f.end(), 0); // 批量递增赋值
        siz.assign(maxn, 1); // 赋值n个1
    }
     
    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
     
    bool same(int x, int y) {
        return find(x) == find(y);
    }
     
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
     
    int size(int x) {
        return siz[find(x)];
    }
};
```



### 拓展域并查集

```C++
/*
https://www.luogu.com.cn/problem/P2024
x为同类域，x+n为捕食域，x+n+n为天敌域
注意：如果x吃y，y吃z，那么z一定吃x，基于这个定理：如果x吃y，那么
a[x] = a[y+n+n] x与y的天敌一定是同类
a[x+n] = a[y] y与x的捕食域中动物一定同类
a[y+n] = a[x] x与y的捕食域中动物一定同类
*/
#include<iostream>
const int N = 5e4 + 10;
int p[N * 4];
int get(int x){
	if(p[x] != x) p[x] = get(p[x]);
	return p[x];
}
void link(int x,int y){
	p[get(x)] = p[get(y)];
}
int main()
{
	int n,k;
	scanf("%d%d",&n,&k);
	for(int i = 1; i <= 3 * n; i ++) p[i] = i;
	
	int ans = 0;
	for(int i = 0; i < k; i ++){
		int op,x,y;
		scanf("%d%d%d",&op,&x,&y);
		if(x > n || y > n || (op == 2 && x == y)){
			ans ++;
			continue;
		}
        //之所以以下的判断，x + n/y还要判断y + n/x不可省略是因为 x与y的捕食与否关系只能是一方对一方，而不可能同时存在，所以两种可能都要判断，其他判断也是这个道理
		if(op == 1){//同类关系
			if(get(x + n) == get(y) || get(y + n) == get(x)) ans ++;//两者存在捕食关系
			else{//关系成立
				//三种域都合并
				link(x,y);//同类
				link(x+n,y+n);//他们两各自的捕食域中的动物也是同类
				link(x+n+n,y+n+n);//天敌域的合并
			}
		}
		else{//x吃y
			if(get(y + n) == get(x) || get(x) == get(y)) ans ++;//是同类或者是y吃x的关系
			else{//关系成立
				link(x+n,y);//x的捕食域与y的同类域合并
				link(y+n+n,x);//y的天敌是x
				link(x+n+n,y+n);//如果x吃y,y吃z,那么z一定吃x即x的天敌域与y的捕食域合并
			}
		}
	}
	printf("%d",ans);
	return 0;
}
```



### 带权并查集

```C++
/*
给出一个数列的长度 N，及 M 个子区间和， 形如：x y z, 表示子区间 [x, y] 的和为 z （-1e12 <= z <= 1e12）
每次询问[l, r]的区间和，若不可得输出no
*/
#include <iostream>
#include <algorithm>
using namespace std;
#define ll long long
const int N = 1e5 + 10;

int n, m, q;
int p[N]; // 对于给出的区间，所有的点的祖宗节点(可以理解为指向)即最左边的点
ll sum[N]; // sum[i]即该点到祖宗节点(最左边点的)距离 带有前缀和的思维

int get(int x){
	if(p[x] != x) {
		int root = p[x];
		p[x] = get(p[x]);
		sum[x] += sum[root]; // 如果父亲有值，说明本次压缩路径找到新的祖先，需要将递归后的父亲的值加起来 
	}
	return p[x];
}
/*
错解 
int get(int x){
	ll s = sum[p[x]];
	if(p[x] != x) {
		int root = get(p[x]);
		sum[x] += sum[p[x]] - s; 
		p[x] = root;
	}
	return p[x];
}
*/
int merge(int l, int r, ll s){
	int rx = get(l), ry = get(r);
	// 因为s可以为负，所以需要确定一个方向为正
	if(rx < ry){
		sum[ry] = sum[l] + s - sum[r];
		p[ry] = rx; 
	}
	else if(rx > ry){
		sum[rx] = sum[r] - sum[l] - s;
		p[rx] = ry;
	}
}

int main(){
	scanf("%d%d%d",&n,&m,&q);
	
	for(int i = 1; i <= n + 1; i ++){ // 注意离散会多一位 0 / n + 1
		p[i] = i;
		sum[i] = 0;
	}
	
	// 因为是数列而不是数轴所以需要离散为左闭右开区间， 或者左开右闭
	for(int i = 1; i <= m; i ++){
		int l, r; ll s;
		scanf("%d%d%lld",&l, &r, &s); r ++;
		merge(l, r, s);
	}
	
	for(int i = 1; i <= q; i ++){
		int l, r;
		scanf("%d%d",&l, &r); r ++;
		if(get(l) != get(r)) printf("UNKNOWN\n");
		else printf("%lld\n",sum[r] - sum[l]);
	}
	return 0;
}

/*
10 3 1
1 1 1
3 3 3
2 2 2
1 3
*/ 
```



## 线段树

### 区间加法+乘法

```C++
#include<iostream>
#define ll long long
const int N = 1e5 + 10;
const int MOD = 571373; 
ll a[N];
struct node
{
	int l,r;
	ll sum,laz1,laz2;//加法懒标记，乘法懒标记 
}tr[N << 2]; 
void pushup(int p){tr[p].sum = (tr[p<<1].sum + tr[p<<1|1].sum) % MOD;}
void build(int p,int l,int r)
{
	tr[p] = {l,r,0,0,1};
	if(l == r){
		tr[p].sum = a[l];
		return ;
	}
	int mid = l + r >> 1;
	build(p<<1,l,mid);
	build(p<<1|1,mid+1,r);
    pushup(p);
}
void pushdown(int p)
{
	ll add = tr[p].laz1;
	ll mul = tr[p].laz2;
	
	//先把孩子的sum*上父亲的乘法标记 再加上父亲的加法标记 
	tr[p<<1].sum = (tr[p<<1].sum * mul % MOD + (tr[p<<1].r - tr[p<<1].l + 1) * add % MOD) % MOD;
	tr[p<<1|1].sum = (tr[p<<1|1].sum * mul % MOD + (tr[p<<1|1].r - tr[p<<1|1].l + 1) * add % MOD) % MOD;
	
	//把孩子的加法标记乘上父亲的乘法标记,再加加法标记 
	tr[p<<1].laz1 = (tr[p<<1].laz1 * mul % MOD + add) % MOD;
	tr[p<<1|1].laz1 = (tr[p<<1|1].laz1 * mul % MOD + add) % MOD;
	
	tr[p<<1].laz2 = (tr[p<<1].laz2 * mul) % MOD;
	tr[p<<1|1].laz2 = (tr[p<<1|1].laz2 * mul) % MOD; 
	
	tr[p].laz1=0, tr[p].laz2=1;
}
void add(int p,int l,int r,ll k)
{
	if(tr[p].l>=l && tr[p].r<=r)
	{
		tr[p].laz1 = (tr[p].laz1 + k) % MOD;
		tr[p].sum = (tr[p].sum + k * (tr[p].r - tr[p].l + 1) % MOD) % MOD;
		return ;
	}
	pushdown(p);
	int mid = tr[p].l + tr[p].r >> 1;
	if(l <= mid) add(p<<1,l,r,k);
	if(r > mid) add(p<<1|1,l,r,k);
	pushup(p);
}

void mul(int p,int l,int r,ll k)
{
	if(tr[p].l>=l && tr[p].r<=r)
	{
		tr[p].sum = tr[p].sum * k % MOD;
		tr[p].laz1 = tr[p].laz1 * k % MOD;
		tr[p].laz2 = tr[p].laz2 * k % MOD;
		return ;
	}
	pushdown(p);
	int mid = tr[p].l + tr[p].r >> 1;
	if(l <= mid) mul(p<<1,l,r,k);
	if(r > mid) mul(p<<1|1,l,r,k);
	pushup(p);
}

ll query(int p,int l,int r)
{
	if(tr[p].l>=l && tr[p].r<=r) return tr[p].sum;
	pushdown(p);
	ll ans=0;
	int mid = tr[p].l + tr[p].r >> 1;
	if(l <= mid) ans = (ans + query(p<<1,l,r)) % MOD;
	if(r > mid) ans = (ans + query(p<<1|1,l,r)) % MOD;
	return ans;
}
int main()
{
	int n,q,mod;
	scanf("%d%d%d",&n,&q,&mod);
	for(int i=1;i<=n;i++)
		scanf("%lld",&a[i]);
	
	build(1,1,n);	
	while(q --){
		ll ans,k;
		int op,l,r;
		scanf("%d%d%d",&op,&l,&r);
		if(op==1){
			scanf("%lld",&k);
			mul(1,l,r,k);
		} 
		else if(op==2){
			scanf("%lld",&k);
			add(1,l,r,k);
		}
		else{
			ans=query(1,l,r);
			printf("%lld\n",ans);
		}
	}
	return 0;
} 
```



### 吉司机势能树

可以实现区间取 $\min/\max$ 的操作，具体的以取 $\max$ 为例。

线段树维护四个值：

- $minv_1$：区间最小值
- $minv_2$：区间严格次小值
- $sum$：区间最小值的个数
- $val$：需要维护的相关的区间值（求和、异或等）

在对区间取 $\max(x)$ 递归到一个待修改区间时，进行如下处理

1. 若 $x\leq minv_1$，本次修改不造成任何影响直接返回。
2. 若 $minv_1<x<minv_2$，本次修改造成的影响可以标记并直接计算出为 $sum*(x-minv_1)$ （以求和为例）
3. 若 $minv_2\leq x$，则直接递归处理

例题：操作：区间 $[l,r]$ 的值对 $x$ 取 $\max$，询问：区间 $[l,r]$ 暂时加入一个新的数 $x$，求区间异或和的最高位 $2^k$ 在区间中的个数。

```C++
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 2e5 + 10;
const int inf = 1 << 30;
struct node{
    int l,r,mnv1,mnv2,laz,sum,val,b[30]; //bi:区间数位2^i的个数
}tr[N * 4];
int a[N];

void pushup(int p){
    node &l = tr[p << 1], &r = tr[p << 1 | 1];
    tr[p].val = l.val ^ r.val;
    
    tr[p].mnv1 = min(l.mnv1, r.mnv1);
    if(l.mnv1 == r.mnv1){
        tr[p].sum = l.sum + r.sum;
        tr[p].mnv2 = min(l.mnv2, r.mnv2);
    }
    else{
        if(l.mnv1 < r.mnv1)  tr[p].sum = l.sum;
        else tr[p].sum = r.sum;
        tr[p].mnv2 = min({max(l.mnv1, r.mnv1), l.mnv2, r.mnv2});
    }
    
    for(int i = 0 ; i < 30; i ++){
        tr[p].b[i] = l.b[i] + r.b[i];
    }
}

void getbit(int p,int x,int k){// 取二进制的每一位
    for(int i = 29; i >= 0; i --){
        if(x >> i & 1) tr[p].b[i] += k;
    }
}

void build(int p,int l,int r){
    tr[p] = {l,r,inf,inf};
    if(l == r){
        tr[p].mnv1 = a[l];
        tr[p].val = a[l];
        tr[p].sum = 1;
        getbit(p, a[l], 1);
        return ;
    }
    int mid = (l + r) >> 1;
    build(p<<1, l, mid);
    build(p<<1|1, mid + 1, r);
    pushup(p);
}

void push(int p, int x){
    if(tr[p].mnv1 >= x) return ;
    if(tr[p].sum & 1) {
        tr[p].val ^= tr[p].mnv1;
        tr[p].val ^= x;
    }
    getbit(p, tr[p].mnv1, -tr[p].sum);
    getbit(p, x, tr[p].sum);
    tr[p].laz = max(tr[p].laz, x);
    tr[p].mnv1 = x;
}

void pushdown(int p){
    if(tr[p].laz){
        push(p<<1,tr[p].laz);
        push(p<<1|1,tr[p].laz);
        tr[p].laz = 0;
    }
}
void update(int p,int l,int r,int k)
{
    if(tr[p].mnv1 >= k) return ;
    if(tr[p].l >= l && tr[p].r <= r && tr[p].mnv2 > k){
        push(p, k);
        return ;
    }
    pushdown(p);
    int mid = (tr[p].l + tr[p].r) >> 1;
    if(l <= mid) update(p<<1,l,r,k);
    if(r > mid) update(p<<1|1,l,r,k);
    pushup(p);
}

int query(int p,int l,int r,int bit){
    if(tr[p].l >= l && tr[p].r <= r){
        return tr[p].b[bit];
    }
    pushdown(p);
    int mid = (tr[p].l + tr[p].r) >> 1, ans = 0;
    if(l <= mid) ans += query(p<<1,l,r,bit);
    if(r > mid) ans += query(p<<1|1,l,r,bit);
    return ans;
}
int query_XOR(int p,int l,int r){
    if(tr[p].l >= l && tr[p].r <= r){
        return tr[p].val;
    }
    pushdown(p);
    int mid = (tr[p].l + tr[p].r) >> 1, ans = 0;
    if(l <= mid) ans ^= query_XOR(p<<1,l,r);
    if(r > mid) ans ^= query_XOR(p<<1|1,l,r);
    return ans;
}
int main()
{
    int n,q;
    scanf("%d%d",&n,&q);
    for(int i = 1; i <= n; i ++){
        scanf("%d",&a[i]);
    }
    build(1,1,n);
    while(q --)
    {
        int op,l,r,x;
        scanf("%d%d%d%d",&op,&l,&r,&x);
        if(op == 1) update(1,l,r,x);
        else{
            int sum_XOR = query_XOR(1,l,r) ^ x, ans = 0;
            if(sum_XOR == 0){
                printf("0\n");
                continue ;
            }
            for(int i = 29; i >= 0; i --){ // 找到区间异或和的最高位
                if((sum_XOR >> i) & 1){
                    ans = query(1,l,r,i);
                    if(x >> i & 1) ans ++;
                    break;
                }
            }
            printf("%d\n",ans);
        }
    }
    return 0;
}
```



### 动态开点线段树

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
const int MAX = 1e9;
#define ll long long
int tot;
ll p;
struct node{
    int a, b;
    bool operator < (const node &A)const{
        return a > A.a;
    }
}s[N];

struct Tree{
    int ls,rs,sum;
}tr[N * 80];

void clear(){
    for(int i = 1; i <= tot; i ++) tr[i] = {0,0,0};
    tot = 0;
}

bool check(ll x, ll y){
    return x * 100 < y * p;
}

void update(int &k,int l,int r,int ql,int qr, int v)
{
    if(!k) k = ++ tot;
    if(ql <= l && qr >= r){
        tr[k].sum += v;
        return ;    
    }
    int mid = (l + r) >> 1;
    if(ql <= mid) update(tr[k].ls, l, mid, ql, qr, v);
    if(qr > mid) update(tr[k].rs, mid + 1, r, ql, qr, v);
    tr[k].sum = tr[tr[k].ls].sum + tr[tr[k].rs].sum;
}

int query(int &k, int l, int r, int loc)
{
    if(!k) return 0;
    if(check(r, loc)){
        return tr[k].sum;
    }
    int mid = (l + r) >> 1, ans = 0;
    ans += query(tr[k].ls, l, mid, loc);
    if(check(mid, loc)) ans += query(tr[k].rs, mid + 1, r, loc);
    return ans;
}

void solve()
{
    int n;
    scanf("%d%lld",&n,&p);
    clear();
    
    int root = 0, maxb = 0;    
    for(int i = 1; i <= n; i ++){
        scanf("%d%d",&s[i].a, &s[i].b);
        update(root, 1, MAX, s[i].a, s[i].a, 1);
        maxb = max(maxb, s[i].b);
    }
    sort(s + 1, s + 1 + n);

    int ans = 0;
    for(int i = 1; i <= n; i ++){
        auto [a, b] = s[i];
        if(a <= maxb) {
            ans = max(ans, n - query(root, 1, MAX, maxb));
            break;
        }
        ans = max(ans, n - query(root, 1, MAX, a));
        update(root, 1, MAX, a, a, -1);
        update(root, 1, MAX, b, b, 1);
    }
    printf("%d",ans);
    return ;
}
int main()
{
    int T;
    scanf("%d",&T);
    for(int i = 1; i <= T; i ++) {
        printf("Case #%d: ",i);
        solve();
        if(i != T) printf("\n");
    }
    return 0;
}
```



### 可持久化线段树

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
struct node{
    int l,r;
    int sum;
    int lth,rth;//左右儿子编号
}tr[N*40];
map<int,int>mp;
int n,m,a[N],fp[N],T[N],cnt;//fp为去重后map映射，T为每次更新后的根节点地址
void build(int p,int l,int r)
{
    tr[p] = {l,r,0,0,0};
    if(l == r) return ;
    tr[p].lth = ++ cnt;
    tr[p].rth = ++ cnt;
    int mid = l + r >> 1;
    build(tr[p].lth,l,mid); 
    build(tr[p].rth,mid+1,r);
}
void pushup(int p)
{
    tr[p].sum = tr[tr[p].lth].sum + tr[tr[p].rth].sum;
}
void update(int tar,int now,int pre)
{
    int l = tr[pre].l, r = tr[pre].r;
    tr[now] = tr[pre];//初始化节点
    if(l == r)
    {
        tr[now].sum ++;
        return ;
    }
    int mid = l + r >> 1;
    if(tar <= mid)
    {
        tr[now].lth = ++ cnt;//需要更新的是左子树，右子树编号保留之前的
        update(tar,tr[now].lth,tr[pre].lth);//继续向下更新左子树
    }
    else
    {
        tr[now].rth = ++ cnt;
        update(tar,tr[now].rth,tr[pre].rth);
    }
    pushup(now);
}
int query(int k,int now,int pre)
{
    int l = tr[now].l, r = tr[pre].r;
    if(l == r)return l;//找到叶子节点即为答案
    int nlth = tr[now].lth;
    int plth = tr[pre].lth;
    int nrth = tr[now].rth;
    int prth = tr[pre].rth;
    int sum = tr[nlth].sum - tr[plth].sum;//此区间左子树数的数量
    if(k <= sum)return query(k,nlth,plth);//进入右子树继续寻找
    else return query(k-sum,nrth,prth);//前sum个不在左子树中 
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++) 
    {
        scanf("%d",&a[i]);
        fp[i] = a[i];
    }

    sort(fp+1,fp+1+n); 
    int len = unique(fp+1,fp+1+n)-(fp+1);//应该减去起始地址，而非固定fp[0]
    for(int i=1;i<=len;i++)mp[fp[i]] = i;//离散化
    
    build(0,1,len);

    for(int i=1;i<=n;i++)
    {
        T[i] = ++ cnt;
        update(mp[a[i]],T[i],T[i-1]);
    }
    while(m -- )
    {
        int l,r,k;
        scanf("%d%d%d",&l,&r,&k);
        printf("%d\n",fp[query(k,T[r],T[l-1])]);
    }
    return 0;
}
```



### 可持久化线段树（建树模板）

```C++
struct node{ // 维护的是多版本的权值树
    int ls, rs, sum;
}tr[N * 40];

int T[N], tot;

void build(int &now, int pre, int loc, int l, int r){
    now = ++ tot;
    tr[now] = tr[pre];
    tr[now].sum ++;
    if(l == r) return ;
    int mid = (l + r) >> 1;
    if(loc <= mid) build(tr[now].ls, tr[pre].ls, loc, l, mid);
    else build(tr[now].rs, tr[pre].rs, loc, mid + 1, r);
}

int query(int lth, int rth, int ql, int qr, int l, int r){ // 询问版本 lth ~ rth 之间 区间 [ql, qr] 的和
    if(l >= ql && r <= qr) return tr[rth].sum - tr[lth].sum;
    int mid = (l + r) >> 1, ans = 0;
    if(ql <= mid) ans += query(tr[lth].ls, tr[rth].ls, ql, qr, l, mid);
    if(qr > mid) ans += query(tr[lth].rs, tr[rth].rs, ql, qr, mid + 1, r);
    return ans;
}

void init(){
    for(int i = 1; i <= tot; i ++) tr[i] = {0, 0, 0};
    for(int i = 1; i <= n; i ++) T[i] = 0;
    tot = 0;
}
```



### 线段树合并

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
struct node{
    int l,r,sum;
}tr[N * 40];
int p[N],id[N],tot;
int get(int x){
    if(x != p[x]) p[x] = get(p[x]);
    return p[x];
}
void pushup(int k){tr[k].sum = tr[tr[k].l].sum + tr[tr[k].r].sum;}
void build(int &k,int l,int r,int loc){
    if(!k) k = ++ tot;
    tr[k].sum = 1;
    if(l == r) return ;
    int mid = (l + r) >> 1;
    if(loc <= mid) build(tr[k].l,l,mid,loc);
    else build(tr[k].r,mid+1,r,loc);
    // pushup(k);
}

int T[N];
void merge(int &u,int v,int l,int r){
    if(!u || !v)u |= v;
    // else if(l == r) return;重要度唯一所以不会递归到根节点
    else{
        int mid = (l + r) >> 1;
        merge(tr[u].l,tr[v].l,l,mid);
        merge(tr[u].r,tr[v].r,mid+1,r);
        pushup(u);
    }
}

int query(int u,int l,int r,int y){
    if(l == r) return l;
    int mid = (l + r) >> 1;
    if(tr[tr[u].l].sum >= y) return query(tr[u].l,l,mid,y);
    else return query(tr[u].r,mid+1,r,y-tr[tr[u].l].sum);
}

int main(){
    int n,m,q,x;
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++){
        p[i] = i;
        scanf("%d",&x);
        id[x] = i;
        build(T[i],1,n,x);
    }

    while(m --){
        int u,v;
        scanf("%d%d",&u,&v);
        u = get(u),v = get(v);
        if(u == v) continue ;
        merge(T[u],T[v],1,n);
        p[v] = u;
    }

    scanf("%d",&q);
    while(q --){
        char op[2];
        int u,v,x,y;
        scanf("%s",op);
        if(*op == 'B'){
            scanf("%d%d",&u,&v);
            u = get(u),v = get(v);
            if(u == v) continue ;
            merge(T[u],T[v],1,n);
            p[v] = u;
        }
        else{
            scanf("%d%d",&x,&y);
            x = get(x);
            if(tr[T[x]].sum < y) puts("-1");
            else printf("%d\n",id[query(T[x],1,n,y)]);
        }
    }
    return 0;
}
```



### 李超线段树

​		一般用于维护一次函数的最值，每次操作为插入一个定义域为 $[l, r]$ 的一次函数，或者询问所有一次函数中在 $x = k$ 处取值最大的那个的编号。

​		对于区间修改的问题，按照一般的方法给每个节点打上懒标记，懒标记表示的是一条线段，记为 $L_i$，表示要用 $L_i$ 更新该节点表示的整个区间（$L_i$ 的定义域至少覆盖整个区间）。对于给一个区间打上懒标记，由于线段之间难以合并我们考虑递归下传。

​		设当前新插入的线段为 $f_i$，递归到每个区间时懒标记为 $g_i$，我们用 $f_i$ 与 $g_i$ 在区间中点处的值作比较。

- 该区间无懒标记，直接打上标记即可。
- 若 $f_i$ 更优， 因为 $g_i$ 的值之前已经递归过了，无论左右子区间 $g_i$ 是否可能更优在此之前都已经更新过了，所以当前区间懒标记直接更新为 $f_i$.
- 若 $g_i$ 更优，则分情况讨论
  1. 若在区间左端点 $l$ 处 $f_i$ 更优，那么 $f_i$ 和 $g_i$ 一定在左子区间产生交点，只有在左子区间 $f_i$ 可能更优，向左子区间递归下传。
  2. 若在区间右端点 $r$ 处 $f_i$ 更优，那么 $f_i$ 和 $g_i$​ 一定在右子区间产生交点，只有在右子区间 $f_i$ 可能更优，向右子区间递归下传。
  3. 若左右端点都是 $g_i$ 更优，则该区间中 $f_i$ 任何位置都不可能成为答案，不需要递归下传。
- 若相等即 $f_i$ 和 $g_i$ 交于中点，可以归于 $g_i$ 更优的情况，使得 $f_i$ 向更优的一端递归下传。

值得注意的是，懒标记并不代表区间所有点最优的线段，因此查询时应该查询所有包括 $x$ 的区间，比较得出最值。

时间复杂度：单次查询 $O(logn)$，插入 $O((logn)^2)$.

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10, mod = 39989, MOD = 1e9, MAX = 40000;
const double eps = 1e-9;

int cmp(double x, double y){
    if(x - y > eps) return 1;
    if(y - x > eps) return -1;
    return 0;
}

struct seg{
    double k, b;
}s[N];

double calc(int id, int x){
    return s[id].k * x + s[id].b;
}

struct seg_tree{
    int ls, rs, id; // 懒标记的是线段对应的id
}tr[N * 4];

int tot;
void upd(int& p, int v, int l, int r){ // 对线段完全覆盖到的所有区间进行更新
    if(!p) p = ++ tot;
    int& u = tr[p].id;
    int mid = (l + r) >> 1;
    int cek = (u > 0) ? cmp(calc(v, mid), calc(u, mid)) : 1;
    if(cek == 1 || (!cek && v < u)) swap(u, v);

    // 若是两线段在区间内相交，则只会有一端更优秀，只会向一边递归
    // 若是区间内完全更优，在本区间直接交换，不会递归
    // 若是区间内完全相等，因为原来的编号更小，也不会递归
    // 以此保证复杂度
    int cl = cmp(calc(v, l), calc(u, l)), cr = cmp(calc(v, r), calc(u, r));
    if(cl == 1 || (!cl && v < u)) upd(tr[p].ls, v, l, mid); 
    if(cr == 1 || (!cr && v < u)) upd(tr[p].rs, v, mid + 1, r);
}

void update(int& p, int ul, int ur, int v, int l = 1, int r = MAX){
    if(!p) p = ++ tot;
    if(ul <= l && r <= ur){
        return upd(p, v, l, r);
    }
    int mid = (l + r) >> 1;
    if(ul <= mid) update(tr[p].ls, ul, ur, v, l, mid);
    if(ur > mid) update(tr[p].rs, ul, ur, v, mid + 1, r);
}

int qmax(int a, int b, int x){
    if(a > b) swap(a, b);
    if(!a || !b) return a | b;
    int cek = cmp(calc(a, x), calc(b, x));
    if(cek >= 0) return a;
    return b; 
}
int query(int p, int x, int l = 1, int r = MAX){
    if(!p || l == r){
    	return tr[p].id;
    }
    int ans = 0, mid = (l + r) >> 1;
    if(l <= x && x <= r) ans = tr[p].id;
    if(x <= mid) ans = qmax(ans, query(tr[p].ls, x, l, mid), x);
    else ans = qmax(ans, query(tr[p].rs, x, mid + 1, r), x);
    return ans;
}

int cnt, lastans, root;
void add(){
    int x0, y0, x1, y1;
    cin >> x0 >> y0 >> x1 >> y1;
    x0 = (x0 + lastans - 1) % mod + 1;
    x1 = (x1 + lastans - 1) % mod + 1;
    y0 = (y0 + lastans - 1) % MOD + 1;
    y1 = (y1 + lastans - 1) % MOD + 1;
	
	if(x0 > x1) swap(x0, x1), swap(y0, y1);
	
    if(x0 == x1){ // 特判垂直
        s[++ cnt] = {0, (double)max(y0, y1)};
    }
    else{
        double k = (double)(y0 - y1) / (double)(x0 - x1);
        double b = (double)y0 - k * x0;
        s[++ cnt] = {k, b};
    }

    update(root, x0, x1, cnt);
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int n;
    cin >> n;
    for(int i = 1; i <= n; i ++){
        int op, k; cin >> op;
        if(op == 1) add();
        else
        {
            cin >> k;
            int x = (k + lastans - 1) % mod + 1;
            lastans = query(root, x);
            cout << lastans << "\n";
        }
    }
    return 0;
}
```

 

## 树状数组

### 一维树状数组

```C++
// 求逆序对
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 5e5 + 10;

int n, a[N], t[N];

struct BIT{
    int maxn, tr[N];
    BIT() {}
    BIT(int len){
        init(len);
    }

    void init(int len){
        maxn = len;
        for(int i = 1; i <= maxn; i ++) tr[i] = 0;
    }

    int lowbit(int x){ return x & -x; }
    void update(int x, int k){
        for(int i = x; i <= maxn; i += lowbit(i)) tr[i] += k;
    }
    int get_pre(int r){
        int ans = 0;
        for(int i = r; i; i -= lowbit(i)) ans += tr[i];
        return ans;
    }
    int get_sum(int l, int r){
        return get_pre(r) - get_pre(l - 1);
    }
};

int main(){
	ios::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	cin >> n;
	for(int i = 1; i <= n; i ++){
		cin >> a[i];
		t[i] = a[i];
	}

	sort(t + 1, t + 1 + n);
	int m = unique(t + 1, t + 1 + n) - (t + 1);
	for(int i = 1; i <= n; i ++){
		a[i] = lower_bound(t + 1, t + 1 + m, a[i]) - t;
	}

	ll ans = 0;
    BIT bit(n);
	for(int i = 1; i <= n; i ++){
		if(a[i] + 1 <= n) ans += bit.get_sum(a[i] + 1, n);
		bit.update(a[i], 1);
	}

	cout << ans << "\n";
	return 0;
}
```



### 二维树状数组

利用二维前缀和的思想。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1010;

struct BIT{
    int maxn, maxm;
    int tr[N][N]; // 二维树状数组

    BIT() {}
    BIT(int lenn, int lenm){
        init(lenn, lenm);
    }

    void init(int lenn, int lenm){
        maxn = lenn; maxm = lenm;
        for(int i = 1; i <= maxn; i ++){
            for(int j = 1; j <= maxm; j ++){
                tr[i][j] = 0;
            }
        }
    }

    int lowbit(int x){ return x & -x; }

    void update(int x, int y, int k){
        for(int i = x; i <= maxn; i += lowbit(i)){
            for(int j = y; j <= maxm; j += lowbit(j)){
                tr[i][j] += k;
            }
        }
    }

    int get_pre(int x, int y){ // 前缀和和
        int ans = 0;
        for(int i = x; i; i -= lowbit(i)){
            for(int j = y; j; j -= lowbit(j)){
                ans += tr[i][j];
            }
        }
        return ans;
    }

    int get_sum(int x1, int y1, int x2, int y2){
        int ans = get_pre(x2, y2);
        ans -= (get_pre(x1 - 1, y2) + get_pre(x2, y1 - 1));
        ans += get_pre(x1 - 1, y1 - 1);
        return ans;
    }
};
```



## 偏序问题

### 二维偏序/二维数点

​		问题通常描述成：在静态序列 $a$ 上，多次询问区间 $[l, r]$ 内大小在 $[x, y]$ 范围内的数的个数，或者：给出平面上的 $n$ 个点的坐标 $p_i(x_i,y_i)$，每次查询 $(a,b,c,d)$，表示求在矩形 $(a,b)$，$(c,d)$ 中的点数。

​		思路：利用前缀和与离线的思想，将每次询问分成两次，询问大小 $[x, y]$ 内的数不变，将区间 $[l, r]$ 变成 $[1, l - 1]$， $[1, r]$。将询问按右端点 $r$ 排序，随着询问 $r$ 变大维护的区间也随之变大（权值树状数组维护 $a_i$），将每次查询到的值减去上一次的即可。

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 10, MAX = 1e6;

struct query{
    int r, x, y, id;
    bool operator < (const query &A)const{
        return r < A.r;
    }
}q[N * 2];

int a[N], ans[N], tr[MAX + 10];

void update(int r){
    for(int i = r; i <= MAX; i += (i & -i)) tr[i] ++;
}

int get_sum(int r){
    int ans = 0;
    for(int i = r; i >= 1; i -= (i & -i)) ans += tr[i];
    return ans;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int n, m;
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    for(int i = 1; i <= m; i ++){
        int l, r, x, y;
        cin >> l >> r >> x >> y;
        q[i] = {l - 1, x, y, i};
        q[i + m] = {r, x, y, i};
        /*
            当以矩阵形式给出时
            cin >> x1 >> y1 >> x2 >> y2;
            x1 ++, y1 ++, x2 ++, y2 ++; // 离散至大于0
            q[i] = {x1 - 1, y1, y2, i};
            q[i + m] = {x2, y1, y2, i};
        */
    }
    m <<= 1;
    sort(q + 1, q + 1 + m);

    for(int i = 1, c = 0; i <= m; i ++){
        auto [r, x, y, id] = q[i];
        while(c < r) update(a[++ c]);
        ans[id] = get_sum(y) - get_sum(x - 1) - ans[id];
        /*
            当以二维点坐标给出时
            auto [x, y1, y2, id] = q[i];
            while(j < n && p[j + 1].x <= x) update(p[++ j].y);
            ans[id] = get_sum(y2) - get_sum(y1 - 1) - ans[id];
        */
    }

    m >>= 1;
    for(int i = 1; i <= m; i ++){
        cout << ans[i] << "\n";
    }
    return 0;
}
```



### 三维偏序/CDQ分治

​		三维偏序问题：有 $n$ 个元素，第 $i$ 个元素有 $a_i,b_i,c_i$ 三个属性，设 $f(i)$ 表示满足 $a_j≤a_i$ 且 $b_j≤b_i$ 且 $c_j≤c_i$ 且 $j\neq i$ 的 $j$ 的数量。对于 $d∈[0,n)$，求 $f(i)=d$ 的数量。

​		cdq分治解法：按 $a_i,b_i,c_i$ 三个关键字排序，此时 $a_i$ 已经是有序的，再用归并排序统计逆序对的方法对于每个 $b_i$ 找到所有满足 $b_j≤b_i$ 的元素，最后将这些元素以 $c_j$ 为下标用树状数组维护统计答案。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10, M = 2e5 + 10;

struct node{
    int a, b, c, v, r;
    bool operator < (const node& A)const{
        if(a != A.a) return a < A.a;
        if(b != A.b) return b < A.b;
        return c < A.c;
    }
    bool operator == (const node& A)const{
        return a == A.a && b == A.b && c == A.c;
    }
}s[N], t[N];

int n, m, k, tr[M], ans[N];

int lowbit(int x) { return x & -x; }
void update(int i, int p){
    for(; i <= k; i += lowbit(i)) tr[i] += p;
}

int query(int i){
    int res = 0;
    for(; i; i -= lowbit(i)) res += tr[i];
    return res;
}

void merge_sort(int l, int r){
    if(l >= r) return ;
    int mid = (l + r) >> 1;
    merge_sort(l, mid);
    merge_sort(mid + 1, r);

    int i = l, j = mid + 1, p = l;
    while(i <= mid && j <= r){
        if(s[i].b <= s[j].b){ // 将满足 bi <= bj 的元素按 ci 为下标存入树状数组
            update(s[i].c, s[i].v);
            t[p ++] = s[i ++];
        }
        else{
            s[j].r += query(s[j].c);
            t[p ++] = s[j ++];
        }
    }
    while(i <= mid) update(s[i].c, s[i].v), t[p ++] = s[i ++];
    while(j <= r) s[j].r += query(s[j].c), t[p ++] = s[j ++];
    
    for(i = l; i <= mid; i ++) update(s[i].c, -s[i].v); // 减去本轮递归的贡献
    for(i = l; i <= r; i ++) s[i] = t[i];
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> k;
    for(int i = 1; i <= n; i ++){
        auto& [a, b, c, v, r] = s[i];
        cin >> a >> b >> c; v = 1;
    }

    sort(s + 1, s + 1 + n);
    m = 0;
    for(int i = 1; i <= n; i ++){ // 去重 （防止归并排序时相同元素之间无法统计）
        if(s[i] == s[i - 1]) s[m].v ++;
        else s[++ m] = s[i];
    }

    merge_sort(1, m);

    for(int i = 1; i <= m; i ++) ans[s[i].r + s[i].v - 1] += s[i].v; // ，每个都要加上相同元素的贡献
    for(int i = 0; i < n; i ++){
        cout << ans[i] << "\n";
    }
    return 0;
}
```



## 树套树

### 树状数组套权值线段树

用于解决动态区间第 $k$ 小，带修前缀和问题的算法。

​		思路：以动态区间第 $k$ 小为例，已知静态区间第 $k$ 小我们使用主席树维护前缀和，但当修改一个数时需要对其后的点都要进行修改，每次修改时间复杂度为 $O(nlogn)$，整体时间复杂度 $O(n^2logn)$. 这里我们考虑用树状数组维护前缀和，修改时可以只修改 $logn$ 个位置，这样时间复杂度降至 $O(nlog^2n)$.

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 5e4 + 10, inf = 2147483647;
int n, m, tot, a[N];

int t[N * 2], len;
struct node{
    int op, l, r, k;
}q[N];

int rt[N]; // 树状数组维护权值树,存储线段树的根， 对于每一个位置上维护一棵前缀权值线段树
struct seg_tree{ // 动态开点权值线段树
    int ls, rs, sum;
}tr[N * 200];


void update(int &p, int loc, int k, int l = 1, int r = len){
    if(!p) p = ++ tot;
    if(l == r) {
        tr[p].sum += k;
        return ;
    }
    int mid = (l + r) >> 1;
    if(loc <= mid) update(tr[p].ls, loc, k, l, mid);
    else update(tr[p].rs, loc, k, mid + 1, r);
    tr[p].sum = tr[tr[p].ls].sum + tr[tr[p].rs].sum;
}

int lowbit(int x){ return x & -x;}
void update(int v, int k){ // 修改树状数组
    for(int i = v; i <= n; i += lowbit(i))
        update(rt[i], a[v], k);
}

int prel[N], prer[N], cnt1, cnt2;
void get_son(int pre[], int cnt, int op){
    if(op == 0) for(int i = 1; i <= cnt; i ++) pre[i] = tr[pre[i]].ls;// 转换为左儿子
    else for(int i = 1; i <= cnt; i ++) pre[i] = tr[pre[i]].rs; // 转换为右儿子   
}

int query_num(int k, int l = 1, int r = len){ // 根据排名查询值，线段树
    if(l == r) return l;
    int mid = (l + r) >> 1, sum = 0;
    for(int i = 1; i <= cnt2; i ++) sum += tr[tr[prer[i]].ls].sum; // 求出[l, r]的左区间的权值和
    for(int i = 1; i <= cnt1; i ++) sum -= tr[tr[prel[i]].ls].sum;

    if(k <= sum){ // 在左区间找答案
        get_son(prel, cnt1, 0);
        get_son(prer, cnt2, 0);
        return query_num(k, l, mid);
    }
    else{ // 在右区间找答案
        get_son(prel, cnt1, 1);
        get_son(prer, cnt2, 1);
        return query_num(k - sum, mid + 1, r);
    }
}
int find_num(int l, int r, int k){ // 根据排名查询值， 树状数组处理出涉及到的区间一次性查询
    // 等同于普通树状数组的pre[r] - pre[l - 1], 根据线段树来计算前缀差
    cnt1 = cnt2 = 0;
    for(int i = l - 1; i; i -= lowbit(i)) prel[++ cnt1] = rt[i];
    for(int i = r; i; i -= lowbit(i)) prer[++ cnt2] = rt[i];
    return query_num(k);
}

int query_rank(int k, int l = 1, int r = len){ // 计算k在指定区间的排名
    if(l == r) return 0;
    int mid = (l + r) >> 1, sum = 0;
    if(k <= mid){
        get_son(prel, cnt1, 0);
        get_son(prer, cnt2, 0);
        return query_rank(k, l, mid);
    }
    else{
        for(int i = 1; i <= cnt2; i ++) sum += tr[tr[prer[i]].ls].sum; // 将左区间的数总量加上
        for(int i = 1; i <= cnt1; i ++) sum -= tr[tr[prel[i]].ls].sum;
        get_son(prel, cnt1, 1);
        get_son(prer, cnt2, 1);
        return sum + query_rank(k, mid + 1, r);
    }
}
int find_rank(int l, int r, int k){
    cnt1 = cnt2 = 0;
    for(int i = l - 1; i; i -= (i & -i)) prel[++ cnt1] = rt[i];
    for(int i = r; i; i -= (i & -i)) prer[++ cnt2] = rt[i];
    return query_rank(k) + 1;
}

int find_prev(int l, int r, int k){ // 寻找k的前驱
    int rank = find_rank(l, r, k) - 1; // 查询k的排名， -1就是前驱的排名
    if(rank <= 0) return 0;
    return find_num(l, r, rank);
}

int find_next(int l, int r, int k){ // 寻找k的后继
    if(k == len) return len + 1;
    int rank = find_rank(l, r, k + 1); // 不能写成 find_rank() + 1  原因1. rank + 1可能也是k不满足后继 原因2. k可能不存在，排在rank的就是k的后继
    if(rank >= r - l + 2) return len + 1;
    return find_num(l, r, rank);
}

void discre(){
    sort(t + 1, t + 1 + len);
    len = unique(t + 1, t + 1 + len) - (t + 1);
    for(int i = 1; i <= n; i ++){
        a[i] = lower_bound(t + 1, t + 1 + len, a[i]) - t;
        update(i, 1);
    }
    for(int i = 1; i <= m; i ++){
        if(q[i].op != 2) q[i].k = lower_bound(t + 1, t + 1 + len, q[i].k) - t; // 除了代表查询排名的k,其他均离散化
    }
    t[0] = -inf;
    t[len + 1] = inf;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> m;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        t[++ len] = a[i];
    }
    for(int i = 1; i <= m; i ++){
        cin >> q[i].op;
        if(q[i].op != 3) cin >> q[i].l >> q[i].r;
        else cin >> q[i].l;
        cin >> q[i].k;
        if(q[i].op >= 3) t[++ len] = q[i].k;
    }
    discre(); // 离散化

    for(int i = 1; i <= m; i ++){
        if(q[i].op == 1){ 
            cout << find_rank(q[i].l, q[i].r, q[i].k) << "\n";
        }
        else if(q[i].op == 2){
            cout << t[find_num(q[i].l, q[i].r, q[i].k)] << "\n";
        }
        else if(q[i].op == 3){ // 修改pos位置上的数为k
            update(q[i].l, -1);
            a[q[i].l] = q[i].k;
            update(q[i].l, 1);
        }
        else if(q[i].op == 4){
            cout << t[find_prev(q[i].l, q[i].r, q[i].k)] << "\n";
        }
        else {
            cout << t[find_next(q[i].l, q[i].r, q[i].k)] << "\n";
        }
    }
    return 0;
}
```



## 树的性质

### 树的重心

​		树的重心：对于一棵无根树，设其中的一个节点作为根，则可以形成一棵有根树。该树以根为分界，分为若干个子树，设其中最大子树具有的节点数为 $siz_{u}$，所有节点里 $siz_{u}$ 值最小的节点就是该树的重心，也叫质心。

**性质：**

1. **以树的重心为根时，所有子树的大小都不超过整棵树大小的一半**即 $siz_{v} <= n / 2$， 此性质与 $O(n(logn)^2)$ 的点分治算法有关。

2. **树中所有点到某个点的距离和中，到重心的距离和是最小的；**如果有两个重心，那么到它们的距离和一样。

3. 把两棵树通过一条边相连得到一棵新的树，那么新的树的重心在连接原来两棵树的重心的路径上。

4. 在一棵树上添加或删除一个叶子，那么它的重心最多只移动一条边的距离。

```C++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
vector<int>g[N];
int n, root, siz[N], ans = N;
void dfs(int u, int fa){
	siz[u] = 1;
	int sum = 0;
	for(int v : g[u]){
		if(v == fa) continue ; 
		dfs(v, u);
		siz[u] += siz[v];
		sum = max(sum, siz[v]); // 记录最大的子树
	}
	sum = max(sum, n - siz[u]); // 以父节点为子树的情况
	if(ans > sum) root = u, ans = sum;
}

int main(){
	scanf("%d",&n);
	for(int i = 1; i < n; i ++){
		int u, v;
		scanf("%d%d",&u, &v);
		g[u].push_back(v);
		g[v].push_back(u);
	}
	dfs(1, 0);
	printf("%d",ans);
	return 0;
}
```



## 树链剖分

### 树链剖分树上操作

```c++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 1e5 + 10;
int mod,tot,cnt,head[N],a[N],old[N];
int dep[N],fa[N],son[N],top[N],dfn[N],siz[N];//通过dfs序重新编号将每种操作变成连续区间，线段树优化
struct node{
    int l,r;
    ll sum,laz;
}tr[N * 4];
struct edge{
    int to,nex;
}e[N * 2];

void add(int from,int to){
    e[++tot].to = to;
    e[tot].nex = head[from];
    head[from] = tot;
}

void dfs1(int u,int Fa)
{
    dep[u] = dep[Fa] + 1;
    fa[u] = Fa, siz[u] = 1;
    son[u] = 0;
    for(int i=head[u]; i ;i=e[i].nex)
    {
        int v = e[i].to;
        if(v==Fa) continue;
        dfs1(v,u);
        siz[u] += siz[v];
        if(siz[v] > siz[son[u]]) son[u] = v;
    }
}
//dfn[i]:转化为的连续区间编号,old[i]:对应的树上编号
void dfs2(int u,int topx)//确定dfs序
{
    top[u] = topx;
    dfn[u] = ++cnt;//保证每条重链在线段树上都是连续的区间 
    old[cnt] = u;
    if(son[u]!=0) dfs2(son[u],topx);
    for(int i=head[u]; i ;i=e[i].nex){
        int v = e[i].to;
        if(v!=fa[u]&&v!=son[u]) dfs2(v,v);
    }
}

void build(int l,int r,int i)
{
    tr[i] = {l,r,0,0};
    if(l==r){
        tr[i].sum = a[old[l]] % mod;
        return ;
    }
    int mid = l+r >> 1;
    build(l,mid,i<<1);
    build(mid+1,r,i<<1|1);
    tr[i].sum = tr[i<<1].sum + tr[i<<1|1].sum;
}

void pushdown(int i)
{
    ll laz = tr[i].laz;
    if(laz){
        tr[i<<1].sum = (tr[i<<1].sum + 1ll * (tr[i<<1].r - tr[i<<1].l + 1) * laz) % mod;
        tr[i<<1|1].sum = (tr[i<<1|1].sum + 1ll * (tr[i<<1|1].r - tr[i<<1|1].l + 1) * laz) % mod;
        tr[i<<1].laz += laz, tr[i<<1|1].laz += laz;
        tr[i].laz = 0;
    }
}

void update(int l,int r,int i,ll k)
{
    if(tr[i].l >= l && tr[i].r <= r){
        tr[i].sum = (tr[i].sum + (tr[i].r - tr[i].l + 1) * k) % mod;
        tr[i].laz += k;
        return ;
    }
    pushdown(i);
    int mid = (tr[i].l + tr[i].r)>>1;
    if(l<=mid) update(l,r,i<<1,k);
    if(r>mid) update(l,r,i<<1|1,k);
    tr[i].sum = tr[i<<1].sum + tr[i<<1|1].sum;
}

ll quer(int l,int r,int i)
{
    if(tr[i].l >= l && tr[i].r <= r) return tr[i].sum;
    pushdown(i);
    ll ans=0;
    int mid = (tr[i].l + tr[i].r) >> 1;
    if(l <= mid) ans = (ans + quer(l,r,i<<1)) % mod;
    if(r > mid) ans = (ans + quer(l,r,i<<1|1)) % mod;
    return ans;
}

void chain1(int u,int v,ll k)//树上u->v路径上所有点权值加k
{
    while(top[u] != top[v])
    {
        if(dep[top[u]] < dep[top[v]])swap(u,v);
        update(dfn[top[u]],dfn[u],1,k);
        u = fa[top[u]];
    }
    if(dep[u] < dep[v]) swap(u,v);
    update(dfn[v],dfn[u],1,k);
}

int chain2(int u,int v)//树上从u->v的路径上权值之和
{
    int ans=0;
    while(top[u]!=top[v])
    {
        if(dep[top[u]]<dep[top[v]])swap(u,v);
        ans = (ans + quer(dfn[top[u]],dfn[u],1))%mod;
        u = fa[top[u]];
    }
    if(dep[u] < dep[v]) swap(u,v);
    ans = (ans + quer(dfn[v],dfn[u],1))%mod;
    return ans;
}

int main()
{
    int n,m,root;
    scanf("%d%d%d%d",&n,&m,&root,&mod);
    for(int i = 1; i <= n; i ++) scanf("%d",&a[i]);
    for(int i = 1; i < n; i ++){
        int u,v;
        scanf("%d%d",&u,&v);
        add(u,v), add(v,u);
    }

    dep[0] = 0;
    dfs1(root,0);
    dfs2(root,root);
    build(1,n,1);

    for(int i = 0; i < m; i ++){
        int u,v,op,k,x,y;
        scanf("%d",&op);
        if(op == 1){
            scanf("%d%d%d",&u,&v,&k);
            chain1(u,v,1ll*k);
        }
        else if(op == 2){
            scanf("%d%d",&u,&v);
            int ans = chain2(u,v);
            printf("%d\n",ans);
        }
        else if(op==3){
            scanf("%d%d",&x,&k);
            update(dfn[x],dfn[x] + siz[x] - 1, 1, 1ll*k);
        }
        else if(op==4){
            scanf("%d",&x);
            int ans = quer(dfn[x],dfn[x] + siz[x] - 1, 1);
            printf("%d\n",ans);
        }
    }
    return 0;
}
```



## 珂朵莉树

珂朵莉树(Old Driver Tree)优雅的暴力
		适用范围：有区间赋值操作，且数据保证随机不会没有区间赋值操作 或者 不会总是在很小的区间赋值时才可以。

​		原理：对某区间进行赋值操作，那么我们就可以将这一段区间浓缩为一个点，以 $[l,r,val]$ 的形式代表该区间，数据随机的情况下，整个区间的点数会降到近似 $log$ 级，在点数很少的情况下所有的询问区间问题都可以以暴力的形式求解。

```C++
#include <bits\stdc++.h>
using namespace std;
#define ll long long
const int N = 1e5 + 10, mod = 1000000007;
struct node
{
    int l,r;
    mutable ll val;//mutable关键字定义一个强制可变量，这样可以使得我们在 set 中修改 val 的值
    bool operator <(const node &A)const {return l < A.l;}
    node(int L,int R,ll Val):l(L),r(R),val(Val){}//构造函数（赋值专用）
    node(int L):l(L){}//构造函数(提取区间专用)
};
set<node>s;
using  si = set<node>::iterator;
/*分裂函数*/
auto split(int pos)//返回set迭代器
{
    if(pos > n) return s.end(); 
    auto it = s.lower_bound(node(pos));//获取第一个左端点不小于pos的结点的迭代器
    if(it != s.end() && it->l == pos) return it;
    it --; //该区间一定包含我们要分裂的点
    ll val = it->val;
    int l = it->l, r = it->r;
    s.erase(it);//删除原来节点 以pos为界分裂为[l,pos-1,val],[pos,r,val]两个点
    s.insert(node(l,pos-1,val));
    return s.insert(node(pos,r,val)).first;
}

/*区间赋值操作 即ODT算法保证复杂度的关键所在*/
void assign(int l,int r,ll val)
{
    auto itr = split(r + 1), itl = split(l);//务必先获取r+1,然后再获取l
    s.erase(itl,itr);//直接删除这两个迭代器之间的所有结点
    s.insert(node(l,r,val));//插入合并的区间
}

ll ksm(ll a,ll b,ll p){
    ll ans = 1;
    while(b){
        if(b & 1) ans = (ans % p) * (a % p) % p;
        a = (a % p) * (a % p) % p;//a = a * a % p 可能a * a就爆ll 了
        b >>= 1;
    }
    return ans % p;
}

void add(int l,int r,int x)
{
    auto itr = split(r + 1), itl = split(l);
    for(auto it = itl;it != itr;it ++) it->val += x;
}
typedef pair<ll,int>PII;
void query(int l,int r,int x)
{
    vector<PII>tmp;
    auto itr = split(r + 1), itl = split(l);
    for(auto it = itl;it != itr;it ++)
        tmp.push_back({it->val,it->r - it->l + 1});
    sort(tmp.begin(),tmp.end());
    int cnt = 0;
    for(PII pi : tmp)
    {
        cnt += pi.second;
        if(cnt >= x) {
            printf("%lld\n",pi.first);
            return ;
        }
    }
}

void query(int l,int r,int x,int y)
{
    auto itr = split(r + 1), itl = split(l);
    ll ans = 0;
    for(auto it = itl;it != itr;it ++) 
        ans = (ans + ksm(it->val,x,y) * 1ll * (it->r - it->l + 1) % y) % y;
    printf("%lld\n",ans);
}

int n,m,a[N];
ll seed,vmax;
int rnd(){
    int ret = seed;
    seed = (seed * 7 + 13) % mod; 
    return ret;
}

int main(){
    scanf("%d%d%lld%lld",&n,&m,&seed,&vmax);
    for(int i=1;i<=n;i++)
    {
        a[i] = (rnd() % vmax) + 1;
        s.insert(node(i,i,a[i]));
    }
    for(int i=1;i<=m;i++)
    {
        int op = (rnd() % 4) + 1;
        int l = (rnd() % n) + 1;
        int r = (rnd() % n) + 1;
        if (l > r) swap(l, r);
        int x,y;
        if (op == 3) x = (rnd() % (r - l + 1)) + 1;
        else x = (rnd() % vmax) + 1;
        if (op == 4) y = (rnd() % vmax) + 1;

        if(op == 1) add(l,r,x);
        else if(op == 2) assign(l,r,x);
        else if(op == 3) query(l,r,x);
        else if(op == 4) query(l,r,x,y); 
    }
    return 0;
}
```



## 莫队算法

### 奇偶分块莫队

时间复杂度：$n\sqrt n$（当指针移动可以 $O(1)$ 求解答案时）。

```C++
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 2e5 + 10, M = 1e6 + 10;

struct quer{
    int l, r, id;
}q[N];

int len;
bool cmp(const quer &A, const quer &B){
    int id1 = A.l / len, id2 = B.l / len;//分块
    if(id1 != id2) return id1 < id2;
    if(id1 & 1) return A.r < B.r;//奇数块向右滚
    else return A.r > B.r;//偶数块向左滚
}

ll moans, ans[N], a[x]; 
void add(int x){
	/* 增加 a[x] 的影响 */
}
void sub(int x){
	/* 删除 a[x] 的影响 */ 
}

int main(){
    int n, m; // 长度为n的a序列， m次询问
    cin >> n >> m;
    for(int i = 1; i <= m; i ++){
        int l, r; cin >> l >> r;
        q[i] = {l, r, i};
    }
    
    len = sqrt(n);
    len = max(len , 1);
    sort(q + 1, q + 1 + m, cmp);

    for(int i = 1, L = 1, R = 0; i <= m ;i ++)
    {
        int l = q[i].l, r = q[i].r;
        while(L < l) sub(L ++);
        while(L > l) add(-- L);
        while(R < r) add(++ R);
        while(R > r) sub(R --);
        ans[q[i].id] = moans;
    }

    for(int i = 1; i <= m; i ++) {
        cout << ans[i] << "\n";
    }
    return 0;
}
```



### 带修莫队

​		莫队是离线算法，因此当题目加入修改操作后一般认为强制在线不可做，但有部分题目时间复杂度允许的情况下能用带修莫队解决。

​		关键点在于再加入一维时间 $t$，一次修改相当于时间变化一次。相比于普通莫队在四种变化基础上 $[l, r + 1]$，$[l, r - 1]$，$[l + 1, r]$，$[l - 1, r]$ 再加入时间 $t$， $[l, r, t - 1]$，$[l, r, t + 1]$。操作上并无太多变化，主要难点还在如何计算分块大小确保时间复杂度上。

​		时间复杂度：设数组大小与询问为同一量级 $n$，修改次数为 $t$，分块大小为 $\sqrt[4]{n^3t}$ 时最优，总体复杂度为 $O(n^{\frac53})$.

```C++
//P1903 [国家集训队] 数颜色 / 维护队列 https://www.luogu.com.cn/problem/P1903
// 1.每次询问区间 [l, r] 数的种类 2.修改第 x 个数为 y
#include <bits/stdc++.h>
using namespace std;

const int N = 140000, S = 1e6 + 10;

int n, m, mq, mc, len; // mq:询问次数 mc 修改次数 len: 分块大小

struct query{
    int l, r, t, id;
}q[N];

struct modify{
    int p, c;
}c[N];

int get(int x){ return x / len; } // 分块
bool cmp(query &A, query &B){
    int al = get(A.l), ar = get(A.r);
    int bl = get(B.l), br = get(B.r);
    if(al != bl) return al < bl;
    if(ar != br) return ar < br;
    return A.t < B.t;
}

int a[N], cnt[S], ans[N], moans;
void add(int x){
    if(++ cnt[x] == 1) moans ++;
}
void sub(int x){
    if(-- cnt[x] == 0) moans --;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];

    moans = mc = mq = 0;
    for(int i = 1; i <= m; i ++){
        string op; int x, y;
        cin >> op >> x >> y;
        if(op[0] == 'Q') q[++ mq] = {x, y, mc, mq};
        else c[++ mc] = {x, y};
    }

    len = cbrt((double)n * max(1, mc)) + 1; // cbrt:根号下3次方
    sort(q + 1, q + 1 + mq, cmp);

    for(int i = 1, L = 1, R = 0, t = 0; i <= mq; i ++){
        auto [l, r, tm, id] = q[i];
        while(R < r) add(a[++ R]);
        while(R > r) sub(a[R --]);
        while(L < l) sub(a[L ++]);
        while(L > l) add(a[-- L]);
        int w1 = t < tm, w2 = t > tm;
        while(t != tm){ // 当前时间不是目标时间
            t += w1;
            if(L <= c[t].p && c[t].p <= R){ // 修改位置在当前所求区间中
                sub(a[c[t].p]);
                add(c[t].c);
            }
            swap(a[c[t].p], c[t].c); // 交换这次修改的值，下次反向移动时能修改回来
            t -= w2;
        }
        ans[id] = moans;
    }

    for(int i = 1; i <= mq; i ++){
        cout << ans[i] << "\n";
    }
    return 0;
}
```



### 回滚莫队/不删除莫队

​		当我们遇到新增操作简单，但删除操作比较困难时，例如当求区间最值时，新增可以 $O(1)$ 比较更新最值但删除时却不能确定最值是否改变。考虑 回滚莫队/不删除莫队，既然维护删除比较困难我们就不维护了。

时间复杂度：$O(n\sqrt n)$

​		证明：分块大小 $len = \sqrt n$ 按左端点分块编号进行排序，再按右端点大小排序，每次询问将询问的左端点在同一块内的预处理出来 $[i, j]$，此时右端点保持升序，分情况处理。

1. 左右端点在同一块内
   ​		暴力求解，直接将 $[l, r]$ 遍历一遍，因为在同一块内复杂度为块的大小 $\sqrt n$ 时间复杂为 $O(n\sqrt n)$

2. 右端点和左端点不在同一块内
   ​		右端点 $r > right$ （块的右端点），右指针 $R$ 不返回的从 $right$ 往右扫，执行 `add(++ R)` 操作，此时我们只处理好了右端点，将当前答案 $res$ 备份 $backup = res$.
   
   ​		对于左端点每次询问左指针 $L$ 都从 $right + 1$ 开始向左扫执行 `add(-- L)`，因为左端点肯定都是在一块内，处理完一次询问后将答案 $res$ 回溯至备份 $backup$ 对于下一次询问重复这个过程。
   
   ​		因为右指针是不返回的往右扫时间复杂度 $O(n)$，左指针每次询问都是重新从 $right$ 出发，因为只在块内进行时间复杂度 $O(\sqrt n)$ 相乘即为这部分时间复杂度。

综上1,2情况的复杂度相同，因此算法复杂度为 $O(n\sqrt n)$.

```C++
// https://www.luogu.com.cn/problem/AT_joisc2014_c
// 事件a_i的重要度为 a_i * Ta（a_i出现的次数）多次询问区间 [l, r] 重要度最大的事件的重要度
#include <bits/stdc++.h>
using namespace std;

#define ll long long
const int N = 1e5 + 10;

struct query{
    int l, r, id;
}q[N];

int n, m, len, a[N], t[N], cnt[N];

int get(int x){ return x / len; } 
bool cmp(query& A, query& B){
    int al = get(A.l), bl = get(B.l);
    if(al != bl) return al < bl;
    return A.r < B.r;
}

void add(int x, ll& res){
    cnt[x] ++;
    res = max(res, 1LL * cnt[x] * t[x]);
}

ll ans[N];
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int n, m;
    cin >> n >> m;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        t[i] = a[i];
    }
    sort(t + 1, t + 1 + n);
    int tn = unique(t + 1, t + 1 + n) - (t + 1);
    for(int i = 1; i <= n; i ++){
        a[i] = lower_bound(t + 1, t + 1 + tn, a[i]) - t;
    }

    for(int i = 1; i <= m; i ++){
        int l, r; 
        cin >> l >> r;
        q[i] = {l, r, i};
    }

    len = sqrt(n);
    sort(q + 1, q + 1 + m, cmp);

    for(int i = 1; i <= m; ){
        int j = i;
        while(j + 1 <= m && get(q[j + 1].l) == get(q[i].l)) j ++; // 寻找在相同块内的

        // 暴力求块内的询问
        int right = get(q[i].l) * len + len - 1; // 确定块内的右边界
        while(i <= j && q[i].r <= right){
            ll res = 0;
            auto [l, r, id] = q[i ++];
            for(int k = l; k <= r; k ++) add(a[k], res); // 直接从 [l, r] 全部计算
            ans[id] = res;
            for(int k = l; k <= r; k ++) cnt[a[k]] --;
        }

        // 处理r在块外的询问
        ll res = 0;
        int R = right, L = R + 1; // 左右指针
        while(i <= j){
            auto [l, r, id] = q[i ++];
            while(R < r) add(a[++ R], res); // 不返回的右移
            ll backup = res; // 备份
            while(L > l) add(a[-- L], res); // 每次都从 right + 1 开始左移
            ans[id] = res;
            
            // 回溯
            while(L < right + 1) cnt[a[L ++]] --;
            res = backup;
        }
        memset(cnt, 0, sizeof cnt);
        // for(int k = right + 1; k <= q[i - 1].r; k ++) cnt[a[k]] = 0; 不同的清空方式
    }

    for(int i = 1; i <= m; i ++){
        cout << ans[i] << "\n";
    }
    return 0;
}
```



### 树上莫队

对于树上路径 $[u, v]$ 的查询可以通过欧拉序/dfs序转化为区间查询。

![](C:\Users\饕餮\Desktop\肖天赐的ACM模板\图片\dfs序.png)

​		dfs序就是根据dfs遍历的顺序写下每个点的编号，第一次遍历写一次，退出时再写一次。例如上图的一种可能的dfs序为：`1 2 2 3 5 5 6 6 7 7 3 4 8 8 4 1`，每个点在dfs序中都出现两次我们将其称为第一次出现 $fist_u$ 和最后一次出现 $last_u$.

我们将询问分情况讨论，查询 $u, v$（默认 $u$ 在dfs序中出现较早）

1. $u$ 是 $v$ 的祖先
   ​		找到 $first_u$ 和 $first_v$ 在此区间只出现一次的即为路径上的点，出现两次说明先访问了无关的子树之后又推出来。

2. $u$ 和 $v$ 的最近公共祖先为 $lca$ 且 $lca \neq u,lca \neq v$ 
   ​		找到 $last_u$ 和 $first_v$ 在此区间只出现一次的即为路径上的点，因为 $u,v$ 在同一棵子树此路径上不会包含最近公共祖先，单独加上即可。

```C++
// https://www.acwing.com/problem/content/2536/ 2534. 树上计数2
// 多次询问树上两点 [u, v] 之间路径上不同点权的种类
#include <bits/stdc++.h>
using namespace std;

const int N = 5e4 + 10, M = 1e5 + 10;

int a[N], t[N];
vector<int> g[N];

int len, ans[M];
struct query{
    int l, r, fat, id;
}q[M];

int get(int x){ return x / len; }
bool cmp(query& A, query& B){
    int al = get(A.l), bl = get(B.l);
    if(al != bl) return al < bl;
    return A.r < B.r;
}

int dep[N], f[N][20], dfn[N * 2], fi[N], la[N], tot;
void dfs(int u, int fa){
    dfn[++ tot] = u;
    fi[u] = tot;

    f[u][0] = fa; 
    dep[u] = dep[fa] + 1;
    for(int i = 1; (1 << i) <= dep[u]; i ++){
        f[u][i] = f[f[u][i - 1]][i - 1];
    }

    for(int v : g[u]){
        if(v != fa) dfs(v, u); 
    }
    dfn[++ tot] = u;
    la[u] = tot;
}

int LCA(int u, int v){
    if(dep[u] < dep[v]) swap(u , v);
    int d = dep[u] - dep[v]; // 记录深度差
    for(int i = 0; i <= 20; i ++){
        if((1 << i) & d) u = f[u][i];// 如果深度差包含2^i 那么就向上跳2^i
    }
    if(u == v) return u; // 如果相同说明lca是u
    for(int i = 19; i >= 0; i --){
        if(f[u][i] != f[v][i]){
            u = f[u][i];
            v = f[v][i];
        }
    }
    return f[u][0];
}

int moans, cnt[N], st[N]; // st:出现次数，出现两次的抵消 cnt:计数
void add(int x){
    st[x] ^= 1;
    if(st[x]) moans += (++ cnt[a[x]] == 1);
    else moans -= (-- cnt[a[x]] == 0);
}

int n, m;
int main(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        t[i] = a[i];
    }
    sort(t + 1, t + 1 + n);
    int tn = unique(t + 1, t + 1 + n) - (t + 1);
    for(int i = 1; i <= n; i ++){
        a[i] = lower_bound(t + 1, t + 1 + tn, a[i]) - t;
    }

    for(int i = 1; i < n; i ++){
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    dfs(1, 0);

    for(int i = 1; i <= m; i ++){
        int u, v;
        cin >> u >> v;
        if(fi[u] > fi[v]) swap(u, v);
        int lca = LCA(u, v);
        if(u == lca) q[i] = {fi[u], fi[v], 0, i};
        else q[i] = {la[u], fi[v], lca, i};
    }

    len = sqrt(n);
    sort(q + 1, q + 1 + m, cmp);

    for(int i = 1, L = 1, R = 0; i <= m ;i ++){
        auto [l, r, fat, id] = q[i];
        while(L < l) add(dfn[L ++]);
        while(L > l) add(dfn[-- L]);
        while(R < r) add(dfn[++ R]);
        while(R > r) add(dfn[R --]);
        bool ad = (fat && !cnt[a[fat]]);
        ans[q[i].id] = moans + ad;
    }
    for(int i = 1; i <= m; i ++) {
        cout << ans[i] << "\n";
    }
    return 0;
}
```



## 启发式合并

​		总数为 $n$ 的若干集合合并，每次随机合并时间复杂度是 $O(n^2)$，若我们每次都将小集合向大集合合并，时间复杂度降为 $O(nlogn)$.



### Dsu on tree

```C++
//https://codeforces.com/contest/600/problem/E
//每个节点有颜色ci，求每个点vi的子树中，数目最多的颜色之和
#include<iostream>
#include<algorithm>
using namespace std;
#define ll long long
const int N = 1e5 + 10;
int n,a[N],head[N],tot;
struct edge{
    int to,nex;
}e[N * 2];
void add(int from,int to)
{
    e[++tot].to = to;
    e[tot].nex = head[from];
    head[from] = tot;
}

int son[N], siz[N];
void dfs(int u,int fa) {
    siz[u] = 1; son[u] = 0;
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(v == fa) continue;
        dfs(v, u);
        siz[u] += siz[v];
        if(siz[son[u]] < siz[v]) son[u] = v;
    }
}
ll ans[N], color;
int cnt[N],maxx,nowson;//cnt：统计当前子树的颜色个数，ans记录答案，color当前子树的颜色之和
void count(int u,int fa,int val){
    /*
        注意这是边遍历子树边将值统计进数组，当需要计算两颗子树之间的问题时，
        需要先计算答案，再统计数组。必要时可以再设立一个val = 0, 代表统计答案，在dfs中遍历u的子节点
        执行 count(0) 计算完当前子树答案再 count(1) 将数据加上
    */

    cnt[a[u]] += val;//当val=1时计算贡献，val=-1时执行删除操作

    /* 不同的题使用不同的方法计算答案 */
    if(cnt[a[u]] > maxx){
        maxx = cnt[a[u]];
        color = a[u];
    }
    else if(cnt[a[u]] == maxx) color += a[u];

    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        //u == nowson continue;的用处，防止在重新计算当前节点轻儿子时将重儿子又计算一次
        if(v==fa || v == nowson) continue;//错误写法：u==son[u]会导致在计算轻儿子子树时，漏计算轻儿子子树中的重儿子子树
        count(v,u,val);
    }
}

void dfs(int u,int fa,bool F)//F=0代表是轻儿子的子树，计算完后需要清除
{
    for(int i = head[u]; i; i = e[i].nex){ //先计算以轻儿子为根的子树
        int v = e[i].to;
        if(v != fa && v != son[u]) dfs(v, u, 0);
    }
    if(son[u]){ // 再计算以重儿子为根的子树
        dfs(son[u],u,1);
        nowson = son[u]; // 记录，防止在重新计算轻儿子时再将此重儿子计算一遍 
    }
    
    count(u, fa, 1); //再计算一次轻儿子将其合并进重儿子，因为有nowson标记，所以不会再计算重儿子
    nowson = 0;//置0,否则会影响下面的清除操作,若不置0，那么就不会将此轻儿子的重儿子清除
    ans[u] = color;
    
    if(!F){ //轻儿子的全部数据清除， 视数据大小情况有时不用再遍历一遍子树而是直接清空计数数组
        count(u,fa,-1);
        maxx = color = 0;
    }
}

int main(){
    scanf("%d",&n);
    for(int i = 1; i <= n; i ++){
        scanf("%d",&a[i]);
    }
    for(int i = 2; i <= n; i ++){
        int u,v;
        scanf("%d%d",&u,&v);
        add(u,v), add(v,u);
    }
    dfs(1, 0);//处理出重儿子
    dfs(1, 0, 1);//dsu on tree
    return 0;
}
```



## 分治

### 点分治

​		点分治，顾名思义就是基于树上的节点进行拆分，对于点的拆分其实就是对于树的拆分。**所以我认为点分治的本质其实是将一棵树拆分成许多棵子树处理，并不断递归这一过程，**这应该也是点分治的精髓。

​		**对于树上路径而言，所有路径都可以由两个点的最近路径来表示。两个点的最近路径又可以由这两个点到他们的最近公共祖先节点的路径之和表示（这里的之和表示的是题目所要统计询问的问题而不单单只是距离权值）。**

算法流程是对于一棵树找重心，将路径分成三种。

1. 经过重心，且端点在子树中。

2. 经过重心，且重心为端点。

3. 不经过重心，路径只在子树内。

**对于路径 1,2，求以重心为根的子树所有节点到根的距离，可以直接计算答案（本质是个暴力的过程）。**

**对于路径 3，则是一个递归的过程，递归到子节点再进行一遍该流程（求重心，遍历所有节点）。**



​		时间复杂度：$O((n + m)logn)$，其中 $n$ 为树的大小，$m$ 视具体求解统计方法而定，对于一般情况以及本题，求解答案时的复杂度在于排序，本题使用的是桶排序 $O(n)$， 于是复杂度为 $O(nlogn)$，若使用其他排序方法则复杂度要多一个 $log$，$O(n(logn)^2)$.

**时间复杂度证明：**

​		首先我们要知道重心的性质：**以树的重心为根时，所有子树的大小都不超过整棵树大小的一半。**

​		设当前树的根为 $u$，也是树的重心，节点数为 $n$ ，当每次我们递归到下一层时，**下一层子树以 $v$ 为根的节点数量至多为 $n/2$，所以最多会递归 $logn$ 层 ，节点数就为 $1$ **.

​		而每层的所有不同子树的节点数之和 $\sum siz_{v} < n$ （因为在计算父亲节点（重心）时，会将父节点去掉），每层的求重心和求距离时间复杂度都是 $O(n)$.

​		最终我们求解答案时要将所有到根的节点的距离排序，使用的是桶排序算法，复杂度 $O(m) = O(n)$. 所以最终的时间复杂度为 $O(nlogn)$.

```C++
// https://www.luogu.com.cn/problem/P3806 
//给定一棵有 n 个点的树，询问树上距离为 k 的点对是否存在。
//对于不同的问题，改变就是get_subtree统计的信息和方式不同，以及get_res计算答案的方式不同
#include <bits/stdc++.h>
using namespace std;

const int N = 1e4 + 5, M = 120, MAX = 1e7;
const int inf = 1 << 30;

int head[N], tot;
struct edge{
    int to, nex, w;
}e[N * 2];

void add(int from,int to,int w){
    e[++ tot].nex = head[from];
    e[tot].w = w;
    e[tot].to = to;
    head[from] = tot;
}

int n, m, q[M], ans[M];
    
bool vis[N];// vis:当前点是否被遍历过

int siz[N], root, min_siz = inf; 
void get_root(int u, int fa, int sum){//找重心 num:整颗子树的大小，每次分治都是新的树，所以树的大小是单独子树的大小
    siz[u] = 1; 
    int max_siz = 0;
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(v == fa || vis[v])continue; 
        get_root(v, u, sum);
        siz[u] += siz[v];
        max_siz = max(max_siz, siz[v]);
    }
    max_siz = max(max_siz, sum - siz[u]);
    if(max_siz < min_siz) min_siz = max_siz, root = u;
}

bool cnt[MAX + 10];  // 保存总统计信息
int sub_cnt[N], num; // 统计每棵子树的信息，具体如何统计视题目而定
void get_subtree(int u, int fa, int dis){ // 暴力统计子树信息
    if(dis <= MAX) sub_cnt[++ num] = dis;
    else return ; // 适当剪枝
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(vis[v] || v == fa) continue;
        get_subtree(v, u, dis + e[i].w);
    }
}

void get_res(int u){ // 对于以u为根的子树，在统计完子树后计算答案
    vector<int> c; // 保存需要清空的数据下标
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(vis[v]) continue;
        num = 0; // 统计新子树对编号置0
        get_subtree(v, u, e[i].w); // 获取 子树所有节点 到 根节点的 信息

        // cnt: 保存之前统计的所有信息，sub_cnt:本次获取的信息 两者结合即为新路径信息
        for(int j = 1; j <= num; j ++){
            for(int k = 0; k <= m; k ++){
                if(q[k] >= sub_cnt[j]) ans[k] |= cnt[q[k] - sub_cnt[j]];
            }
        }

        for(int j = 1; j <= num; j ++){ // 将本次统计子树信息记录进总信息中
            if(!cnt[sub_cnt[j]]) c.push_back(sub_cnt[j]);
            cnt[sub_cnt[j]] = 1;
        }
    }
    // 当统计完以 u 为 根的答案时需要清空总统计数组
    for(auto idx : c) cnt[idx] = 0;
}

void Divide(int u){ // 每次传入的 u 都是子树的重心
    vis[u] = cnt[0] = 1;
    get_res(u); // 求解以 u 根的子树
    for(int i = head[u]; i; i = e[i].nex){
        int v = e[i].to;
        if(vis[v]) continue;    // 因为每次是找重心为根所以不能只以!=fa防止遍历到父亲节点，其余vis同理
        min_siz = inf, root = 0;
        get_root(v, 0, siz[v]); // 寻找子树的重心
        Divide(root);           // 进行递归求解子树
    }
}

int main(){
    cin >> n >> m;
    for(int i = 1; i < n; i ++){
        int u, v, w;
        cin >> u >> v >> w;
        add(u, v, w);
        add(v, u, w);
    }

    for(int i = 1; i <= m; i ++) cin >> q[i]; // 储存答案离线查询

    get_root(1, 0, n); // 每次先找重心
    Divide(root);      // 点分治

    for(int i = 1; i <= m; i ++){
        if(ans[i]) cout << "AYE\n";
        else cout << "NAY\n";
    }
    return 0;
}
```



### 根号分治

​		一般而言是将数据以 $\sqrt n$  为界分成两部分，分别用两者时间上的优势求解，下面用例题进行讲解。（有时以 $n^{\frac13}$ 为界更优）

例题一：给定长度为 $n$ 数组 $val$， $m$ 次操作，每次修改 $val_i$ 为 $x$ 或询问  $\sum_1^n val_i  (i\mod x = y)$ .

1. 询问的 $x \leqslant \sqrt n$
   ​		预处理出 $sum_{x,y} = \sum_1^nval_i(i \mod x = y)$，预处理时间复杂度 $n\sqrt n$，之后每次修改都需要 $\sqrt n$，询问 $O(1)$，总复杂度 $O((n + m)\sqrt n)$.

2. 询问的 $x > \sqrt n$ 
   ​		每次直接暴力求解，因为模数 $x$ 足够大，数组中与 $y$ 同余的数不超过 $\sqrt n$，时间复杂度 $O(m\sqrt n)$.

​		综上，若是不对 $x$ 的大小以 $\sqrt n$ 进行区分，单独任一解法复杂度都是 $n^2$ 级别，进行区分后总复杂度降至 $O(n\sqrt n)$ （默认$n,m$ 同阶）。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 2e5 + 10, M = 500;

int n, m, len, val[N], sum[500][500]; // sumij:下标模i余j的val之和

void update(int x, int y){
    for(int j = 1; j <= len; j ++){ // sqrt(n) 时间修改
        sum[j][x % j] += y - val[x];
    }
    val[x] = y;
}

int get_sum(int x, int y){
    int ans = 0;
    if(x > len){
        for(int r = (y == 0?x:y); r <= n; r += x) ans += val[r];
        return ans;
    }
    return sum[x][y];
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> val[i];
    len = cbrt(n); // 开三次方

    for(int i = 1; i <= len; i ++){ // 枚举模数，预处理sum[i][j]
        for(int j = 1; j <= n; j ++){
            sum[i][j % i] += val[j];
        }
    }

    for(int i = 1; i <= m; i ++){
        string op; int x, y;
        cin >> op >> x >> y;
        if(op[0] == 'C') update(x, y);
        else cout << get_sum(x, y) << "\n";
    }
    return 0;
}
```



### 整体二分

​		比赛中有一部分题目可以使用二分的办法来解决。但是当这种题目有多次询问且我们每次查询都直接二分可能导致 TLE 时，就会用到整体二分。整体二分的主体思路就是把多个查询一起解决。这是一种离线算法（一些数据结构题的非经典解法）。

使用整体二分需要满足以下性质：

1. 询问的答案具有可二分性
2. **修改对判定答案的贡献互相独立**，修改之间互不影响效果
3. 修改如果对判定答案有贡献，则贡献为一确定的与判定标准无关的值
4. 贡献满足交换律，结合律，具有可加性
5. 题目允许使用离线算法



​		题目的二分过程一般为 $sol(ql, qr, L, R)$，其中 $[ql,qr]$ 代表询问的问题编号的区间，$[L,R]$ 代表问题答案所在区间。

- 将问题和操作按顺序存入数组中，然后开始分治，此时所有问题的答案都是 $[val_{min},val_{max}]$ 即二分的开始。
- 在每层分治中，利用数据结构（通常是树状数组）来维护当前查询数据。
- 将当前层的答案分为 $[L,mid],[mid+1,R]$ 左右两部分，用维护好的数据来对当前区间问题依次查询，即传统二分的 check 部分，将问题以 check 的结果分成左右部分，代表答案分别落在左右区间。
- 继续向下分治，直到 $L = R$，此时区间 $[ql,qr]$ 的问题答案都是 $L$.

​		时间复杂度：我们每次分治将答案 $[val_{min},val_{max}]$ 分成两个部分，这样的划分会进行 $logval_{max}$ 次，一次划分需要维护的数据是整个数组，设维护数据的时间复杂度为 $O(T)$，则总时间复杂度为 $O(Tlogn)$.

算法基本模板

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 2e5 + 10, maxq = 1e5 + 10;

struct Query{
    int k, id; // 询问，询问的编号
}q[maxq], q1[maxq], q2[maxq];

int ans[N];

bool check(){}
void solve(int ql, int qr, int L, int R){
    if(ql > qr) return ;
    if(L == R){
        for(int i = ql; i <= qr; i ++){
            ans[q[i].id] = L;
        }
        return ;
    }

    int mid = (L + R) >> 1;

    // 维护数据 update(x, 1)

    int cntl = 0, cntr = 0;
    for(int i = ql; i <= qr; i ++){
        if(check()){
            q1[++ cntl] = q[i]; // 临时存答案落在左区间的问题 
        }
        else{
            q2[++ cntr] = q[i]; // 答案落在右区间的问题
        }
    }

    // 清除数据 update(x, -1)

    // 将问题按答案落在区间左右重新存入数组
    for(int i = 1; i <= cntl; i ++) q[ql + i - 1] = q1[i]; // 答案落在左边的问题排在左边
    for(int i = 1; i <= cntr; i ++) q[ql + cntl - 1 + i] = q2[i];

    // 递归处理
    solve(ql, ql + cntl - 1, L, mid);
    solve(ql + cntl, qr, mid + 1, R);
}
```

 

例题：大小为 $n*n$ 的矩阵，$q$ 个询问，每次查询一个子矩阵的第 $k$ 小值。$1\leq n\leq 500,1\leq q\leq 60000$.

​		时间复杂度：所有数据大小是 $m = n * n$，分治的每一层树状数组维护数据的时间复杂度为 $O(mlogm)$，总共分治 $logm$ 次，总时间复杂度为 $O(m(logm)^2)$.

```C++
// P1527 [国家集训队] 矩阵乘法 https://www.luogu.com.cn/problem/P1527
#include <bits/stdc++.h>
using namespace std;

const int N = 510, maxq = 60010;

int n, m, ans[maxq];

struct Query{
    int x1, y1, x2, y2, k, id;
}q[maxq], q1[maxq], q2[maxq];

struct node{
    int x, y, w;
    bool operator < (const node& A)const{
        return w < A.w;
    }
}s[N * N];

struct BIT{
    int maxn, maxm;
    int tr[N][N]; // 二维树状数组

    BIT() {}
    BIT(int lenn, int lenm){
        init(lenn, lenm);
    }

    void init(int lenn, int lenm){
        maxn = lenn; maxm = lenm;
        for(int i = 1; i <= maxn; i ++){
            for(int j = 1; j <= maxm; j ++){
                tr[i][j] = 0;
            }
        }
    }

    int lowbit(int x){ return x & -x; }

    void update(int x, int y, int k){
        for(int i = x; i <= maxn; i += lowbit(i)){
            for(int j = y; j <= maxm; j += lowbit(j)){
                tr[i][j] += k;
            }
        }
    }

    int get_pre(int x, int y){ // 前缀和和
        int ans = 0;
        for(int i = x; i; i -= lowbit(i)){
            for(int j = y; j; j -= lowbit(j)){
                ans += tr[i][j];
            }
        }
        return ans;
    }

    int get_sum(int x1, int y1, int x2, int y2){
        int ans = get_pre(x2, y2);
        ans -= (get_pre(x1 - 1, y2) + get_pre(x2, y1 - 1));
        ans += get_pre(x1 - 1, y1 - 1);
        return ans;
    }
};

BIT bit; // tr[i][j] = 1 即说明对应的二维数组 a[i][j] 的值在此二分出来的值域[L, mid]中
void solve(int ql, int qr, int L, int R){ // ql,qr:问题编号的区间，[L,R]:二分的答案区间 
    if(ql > qr) return ;
    if(L == R){ // 二分找到答案
        for(int i = ql; i <= qr; i ++) {
        	ans[q[i].id] = s[L].w; // 二分的是值的排名
        }
        return ;
    }

    int mid = (L + R) >> 1;
    for(int i = L; i <= mid; i ++){ // 将二分的左半值域对应的二维数组插入树状数组维护
        bit.update(s[i].x, s[i].y, 1);
    }

    int cntl = 0, cntr = 0;
    for(int i = ql; i <= qr; i ++){
        int sum = bit.get_sum(q[i].x1, q[i].y1, q[i].x2, q[i].y2);
        if(sum >= q[i].k){ // 说明问题qi的答案在二分的值域的左区间
            q1[++ cntl] = q[i];
        }
        else{ // 说明在右区间
            q2[++ cntr] = q[i];
            q2[cntr].k -= sum; // 将处在左区间的减去，重新开始算排名
        }
    }

    for(int i = L; i <= mid; i ++){ // 清空本次二分的影响
        bit.update(s[i].x, s[i].y, -1);
    }

    for(int i = 1; i <= cntl; i ++) q[ql + i - 1] = q1[i]; // 答案落在左边的问题排在左边
    for(int i = 1; i <= cntr; i ++) q[ql + cntl - 1 + i] = q2[i];

    // 递归处理
    solve(ql, ql + cntl - 1, L, mid);
    solve(ql + cntl, qr, mid + 1, R);
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cin >> n >> m;
    int tot = 0;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= n; j ++){
            s[tot + 1] = {i, j, 0};
            cin >> s[++ tot].w;
        }
    }
    sort(s + 1, s + 1 + n * n); // 对矩阵元素按值排序

    for(int i = 1; i <= m; i ++){
        auto& [x1, y1, x2, y2, k, id] = q[i];
        cin >> x1 >> y1 >> x2 >> y2 >> k;
        id = i;
    }

    bit.init(n, n);
    solve(1, m, 1, n * n);

    for(int i = 1; i <= m; i ++) cout << ans[i] << "\n";
    return 0;
}
```



# 动态规划



## 区间DP

区间dp的通用模板

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 510;
int f[N][N], val[N];
int main(){
    int n; cin >> n;
    for(int i = 1; i <= n; i ++) f[i][i] = val;     // 预处理长度为1区间
    for(int i = 1; i <= 2 * n; i ++) f[i][i] = val; // 当遇到环形时，预处理出2*n的区间，在2*n的区间上求解
    
    for(int i = 2; i <= n; i ++){                   // 枚举区间长度
        for(int l = 1; l + i - 1 <= n; l ++){       // 枚举区间左端点
            int r = l + i - 1;                      // 区间右端点
            for(int k = l; k < r; k ++){            // 枚举区间合并的分隔点， 有时区间合并并不一定需要枚举分隔点
                f[l][r] = f[l][k] + f[k + 1][r];    // 其中 '+' 为适当的区间转移方程
            }
        }
    }
    cout << f[1][n];
    return 0;
}
```



## 数位DP

​		除了 $pos,lim$ 外的状态，需要考虑时空复杂度，以及题目的性质，是否要作为记忆化的一维。

​		关于 $lim$ 是否要作为记忆化一维的问题，在有多组询问或者以前缀和思想求区间的题目中，需要多次对不同的数进行数位dp，不同的数有不同的上界，所以不能将 $lim$ 来作为记忆化的一维。而单次询问（例题3）则可以作为一维来大幅减少时间复杂度（哪怕数量较少的多次询问（例题2），若将 $lim$ 作为一维也会导致需要每次重置数组，以及降低复用性，不建议采取）。

例题1：$f(x):$ $x$ 的数位和，多次询问 $\sum_L^Rf(x)$.

```C++
#include <bits/stdc++.h>
using namespace std;
#define ll long long
const int N = 20, mod = 1e9 + 7;

int num[N], f[N][200]; // i 位数字，初始数位和为 j 的总数位和 

int dfs(int pos, int lim, int sum){ // pos: 数位 lim：是否到达上下界
	if(pos <= 0) return sum;
	if(!lim && f[pos][sum] != -1) return f[pos][sum]; // 未到上界，且已经递归过
	
	int res = 0;
	int upper = lim ? num[pos] : 9;
	for(int i = 0; i <= upper; i ++){
		res = (res + dfs(pos - 1, lim & (i == upper), sum + i)) % mod;
	}
	if(!lim) f[pos][sum] = res;
	return res;
}

int dp(ll x){
	int cnt = 0;
	while(x){
		num[++ cnt] = x % 10; x /= 10; // 从高位到低位
	}
	return dfs(cnt, 1, 0);
}

int get_ans(ll L, ll R){
	return (dp(R) - dp(L - 1) + mod) % mod; // 前缀和
}

int main(){
	ios::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	memset(f, -1, sizeof f); // 不宜用0作为未递归的状态，可能存在多个状态答案为0，错判为未递归
	int t;
	cin >> t;
	while(t --){
		ll L, R;
		cin >> L >> R;
		cout << get_ans(L, R) << "\n";
	}
	return 0;
}
```



例题2：$f(x):x$ 数位中不同数字的个数，求 $\sum_L^Rf(x)$，$L\leq R\leq2*10^{10^5}$.

​		考虑dp方程 $f[pos][state]:$ 其中 $state$ 代表二进制状态压缩的 $0\sim9,10$ 个数字的存在状态。这样空间复杂度就达到了 $O(|x|2^{10})$，其中 $|x|$ 为位数，这样的空间复杂度显然是我们不能接受的。考虑时间换空间，第二维状态只记录不同的数的个数，具体有的数的状态递归暴力计算。  

​		为什么当 $lim$ 未达上界，数位 $pos$，和已经有的数的种类个数 $cnt$ 确定时就可以判定为搜索到了相同状态可以返回，而不用考虑 $state$ 不同？因为此时后续的剩余数位数量相同，未达上界可以任意填的情况下最终 $cnt = A$ 合法的情况肯定也是相同的（相当于转化为填已有 $cnt$ 种数不计，最终这 $pos$ 位上填的数的种类 $+cnt = A$ 的方案数），所以不用考虑这 $cnt$ 种数具体数字种类状态。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 2e5 + 10, mod = 1e9 + 7;

string L, R;

int f[N][11]; // 长度为i， 函数值为j
int n, A, num[N];

int dfs(int pos, int state, int lim){
    int cnt = __builtin_popcount(state);
    if(pos <= 0) return (cnt == A); // 只记录有多少不同数，不记录数的具体状态，牺牲时间换空间
    if(!lim && f[pos][cnt] != -1) return f[pos][cnt];

    int res = 0;
    int upper = lim ? num[pos] : 9;
    for(int i = 0; i <= upper; i ++){
        res = (res + dfs(pos - 1, state | (1 << i), lim & (i == upper))) % mod;
    }
    if(!lim) f[pos][cnt] = res;
    return res;
}

int dp(string& s){
    int cnt = 0;
    for(int i = n - 1; i >= 0; i --){
        num[++ cnt] = s[i] - '0';
    }
    return dfs(n, 0, 1);
}

int get_val(string& s){
    int cnt[11] = {0}, ans = 0;
    for(auto ch : s){
        if(!cnt[ch - '0']){
            cnt[ch - '0'] = 1;
            ans ++;
        }
    }
    return ans == A;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    
    memset(f, -1, sizeof f);

    cin >> n >> L >> R >> A;
    
    int l = dp(L);
    int r = dp(R) + get_val(L);

    int ans = (r + mod - l) % mod;
    cout << ans << "\n";
    return 0;
}
```



例题3：$S(x):$ $x$ 的数位和，求 $1\sim n$ 中满足 $A<B,S(A)>S(B)$ 的数对 $A, B$ 数量，$n\leq10^{100}$.

​		不同于其他数位dp只对单一数字计数，这里要统计的是数对。我们同样从中提取要维护的关键状态，同时枚举 $A,B$，考虑简化状态，我们不关心具体数字是多少，只关心 $S(x)$ 函数的相对大小，即维护两者差值。为了保证 $A<B$，再维护一个两者是否完全相等的状态。

​		本题是特殊的无多组，也不是询问区间，不会对不同数进行数位dp的题目，所有上界都相同可以将 $lim$ 状态作为一维加入，可以大幅降低时间复杂度。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 110, M = 1010, mod = 1e9 + 7;

int num[N], f[N][M * 2][2][2]; // 直接枚举A, B 之间的差值

int dfs(int pos, int gap, int sam, int lim){ // 数位， 枚举的B - A之间的差值， A,B是否完全相同，B是否到达上界
    if(pos <= 0) return gap < 1000; // 差值存在负数，离散1000为0
    if(f[pos][gap][sam][lim] != -1) return f[pos][gap][sam][lim];

    int res = 0;
    int upper = lim ? num[pos] : 9; // B 的上界
    int up = sam ? upper : 9; // 若 A = B 则两者上界相同，否则A上界为9
    for(int i = 0; i <= up; i ++){ // 枚举 A
        for(int j = sam?i:0; j <= upper; j ++){ // 枚举 B， 若AB完全相同，由于B > A，则数位B必须大于等于A B_pos >= A_pos;
            res = (res + dfs(pos - 1, gap + j - i, sam & (i == j), lim & (j == upper))) % mod;
        }
    }
    return f[pos][gap][sam][lim] = res;
}

int dp(string& s){
    int cnt = s.length();
    for(int i = 0; i < cnt; i ++){
        num[cnt - i] = s[i] - '0';
    }
    return dfs(cnt, 1000, 1, 1);
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    string s; 
    cin >> s;

    memset(f, -1, sizeof f);
    cout << dp(s);
    return 0;
}
```



## 决策单调性优化DP

​		决策单调性，是在**最优化dp**中的可能出现的一种性质，利用它我们可以降低转移的复杂度。最优化dp就是每个状态都是由某一个最优的状态转移而来，通常出现在求最大最小值的dp中，我们称这个最优状态为最优转移点。

**定义：**对于形如 $f_i = min_{0≤j<i}(f_j + w(j, i))$ 的状态转移方程，记 $p_i$ 为 $f_i$ 取到最优值时 $j$，此时 $p_i$ 即为 $f_i$ 的最优决策。**如果 $p_i$ 在 $[1, n]$ 上单调不减，则称 $f$ 函数具有决策单调性**。

**定理1：**对于形如 $f_i = min_{0≤j<i}(f_j + w(j, i))$ 的状态转移方程，若函数 $w$ 满足四边形不等式，则称 $f$ 函数具有决策单调性。

​		一般大多数决策单调性优化dp都是将四边形不等式性质隐藏在普通dp转移方程中，通常需要去发现并证明，或者用打表的方式发现规律。

### 四边形不等式

**定义：**假设我们有 $p_1≤p_2≤p_3≤p_4$。且 $w(p_1,p_3) + w(p_2,p_4) ≤ w(p_1,p_4) + w(p_2,p_3)$ 称函数 $w$ 满足四边形不等式，使用图形化的语言就是 交叉 $≤$ 包含。

使用反证法证明定理1：

设 $f_i$ 的最优决策点为 $p_i$，$y < x$ 且 $p_x < p_y$，根据最优决策点定义：

​						$f_x = f_{p_x} + w(p_x, x) ≤ f_{p_y} + w(p_y,x)$			(1)

此时我们有 $p_x < p_y < y < x$，由四边形不等式得：

​						$w(p_x,y) + w(p_y,x) ≤ w(p_x,x)+w(p_y,y)$   (2)

将(1)(2)相加可得： 

​						$f_{p_x} + w(p_x,y) ≤ f_{p_y} + w(p_y,y)$					    (3)

​																																			

于是发现 $f_y$ 的最优点变成 $p_x$，与定义矛盾，所以根据以上证明可以发现 $y < x$，$p_y < p_x$ 最优决策点单调不减，满足定义。



常用的优化方式之一：**二分单调队列**

​		假设当前节点为 $f_k$，有两个决策点 $i,j(i<j)$，此时 $p_k = i$.

​		随着 $k$ 增大 $k'$，当 $j$ 比 $i$ 更优时，此后的最优决策点 $p_{k'} ≥ j$，即决策单调性的定义：最优决策点单调不减。

​		用更形象的方式表示，假设最开始时所有节点的最优决策点序列为 $0$，$p = {0,0,0,0,\dots,0,0}$，当考虑以 $1$ 为决策点时会将一部分的之前最优决策点覆盖变成 $p={0,0,1,1\dots,1,1}$，以此类推最终不断覆盖形成最终的最优决策点序列 $p={0,0,1,2,3,\dots,p_n}$.

​		因此我们发现每个决策点满足二分性，每次的新决策点都要找到能覆盖之前决策点的维护区间的最左端，这个过程可以用二分查询，而当前最优的决策点存在队首中，值得注意的是此方法我们应该保证能快速求得 $w(l, r)$，一般为 $O(1)$ 或 $O(logn)$.

​		$pre$ 代表队列中前一个节点，$now$ 代表当前节点，有如下几种情况，其中 $l[i],r[i]$ 代表 $i$ 决策点维护的区间。

1. 当前节点维护区间完全覆盖之前的区间（$w(l[i],now)<w(l[i],pre)$）

   直接将 $pre$ pop 出队列舍弃，再去二分寻找正确的左端点。

2. 当前节点维护区间不能覆盖之前区间任意节点 $w(pre,n)<w(now,n)$

   直接舍弃当前节点，继续维护之前的节点。

例题：给定 $n$ 个字符串，按顺序对其进行排版（可排成多行），每行的代价为 $\vert len - L\vert ^P$（$len$ 为该行的总长度），并且同一行两个不同字符串之间需要空格，问如何排版代价总和最小。

```C++
// https://www.luogu.com.cn/problem/P1912
/*
f[i]:前i句排版的最小代价
转移时: f[i] = min_{0<=k<i}f[k] + w(k + 1, i)
其中 w(l, r) 为将[l, r] 排在一行的代价
*/
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define LD long double

const int N = 1e5 + 10;
const ll inf = 1e18;

string s[N];
int n, L, P;

LD f[N]; // 前i句诗排版的最小代价
int pre[N], p[N]; // pi:fi的最优决策点

LD ksm(LD a, int b = P){
    LD res = 1;
    while(b){
        if(b & 1) res = res * a;
        a = a * a;
        b >>= 1;
    }
    return res;
}

LD w(int l, int r){
    return f[l] + ksm(abs(pre[r] - pre[l] + r - l - 1 - L));
}
void output();

int q[N], l[N], r[N], tot, top; // q:维护的队列，l, r对应决策i的维护区间
void init(){
    for(int i = 1; i <= n; i ++){
        l[i] = r[i] = p[i] = 0;
        pre[i] = pre[i - 1] + s[i].size();
    }
}

int binary(int x){ // 在之前决策点维护的范围中 二分 求当前决策点的左端点能覆盖（维护）到节点（比之前节点更优的最左边） 
    int lth = q[tot], ls = l[q[tot]], rs = n;
    while(ls <= rs){
        int mid = (ls + rs) >> 1;
        if(w(x, mid) <= w(lth, mid)) rs = mid - 1;
        else ls = mid + 1;
    }
    return ls;
}

void solve(){
    cin >> n >> L >> P;
    for(int i = 1; i <= n; i ++) cin >> s[i];
    
    init();

    tot = 1, top = 1;
    q[1] = 0, l[0] = 1, r[0] = n;
    for(int i = 1; i <= n; i ++){
        while(top <= tot && r[q[top]] < i) ++ top; // 队首的决策点覆盖不到当前节点，舍弃
        p[i] = q[top];
        f[i] = w(q[top], i);
        if(w(q[tot], n) < w(i, n)) continue ; // 当前决策点过于劣无法覆盖之前的决策点，直接舍弃
        while(tot >= top && w(i, l[q[tot]]) < w(q[tot], l[q[tot]])) -- tot; // 当前决策点完全覆盖之前的决策点，舍弃之前的
       
        int ls = binary(i);
        r[q[tot]] = ls - 1;// 更新之前决策点的左端点（已经被当前节点覆盖）
        l[i] = ls, r[i] = n; // 将当前节点加入队列维护
        q[++ tot] = i;
    }
    
    output();
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int t;
    cin >> t;
    while(t --){
        solve();
    }
    return 0;
}

void output(){
    if(f[n] > inf){
        cout << "Too hard to arrange\n";
        cout << "--------------------\n";
        return ;
    }

    cout << (ll)(f[n] + 0.5) << "\n";
    vector<pair<int, int> > v;
    int id = n;
    while(id){
        v.push_back({p[id] + 1, id});
        id = p[id];
    }
    reverse(v.begin(), v.end());
    for(auto [l, r] : v){
        for(int i = l; i <= r; i ++){
            cout << s[i];
            if(i != r) cout << " ";
        }
        cout << "\n";
    }
    cout << "--------------------\n";
}
```



常用的优化方式之二：**分治**

​		适用于特殊二维dp，只有上一层向下转移，同一层之间不会转移，并且每一层中有决策单调性。一般的做法是每个状态都枚举前一层所有转态，时间复杂度 $O(n^2k)$ （$k$ 是层数）.

​		由于同一层没有转移，所以同一层中我们先转移哪一个位置是没影响的。考虑分治，可以先转移一层中的 $mid$ 状态得到最优决策点为 $p$，之后就能知道 $[l,mid]$ 和 $[mid+1,r]$ 候选转移区间分别为 $[1,p]$ 和 $[p,r]$.这样复杂度降低到 $O(knlogn)$.



### 二维决策单调性优化DP

接下来我们将一维决策单调性拓展到二维，可以应用于一些区间dp中。

**包含单调**

**定义：**$p_1≤p_2≤p_3≤p_4$ 且 $w(p_2,p_3) ≤ w(p_1,p_4)$ 称之为包含单调。

**定理2：**在状态转移方程 $f[i][j] = min_{i≤k<j}f[i][k] + f[k + 1][j] + w(i,j)$ （特别的，$f[i][i] = w(i,i) = 0$）中，如果 $w$ 满足四边形不等式且也满足包含单调，那么$f$ 也满足四边形不等式。

**定理3：** 如果状态转移方程 $f$ 也满足四边形不等式，那么 $∀i<j,p[i][j-1] ≤ p[i][j]≤p[i + 1][j]$.



例题：数轴上 $n$ 个点，任选 $m$ 个点建邮局，使得每个点和最近邮局之间距离和最小，距离为两点坐标绝对值。

dp 定义如下

$x[i]$：第 $i$ 个村庄在数轴上的坐标。

$f[i][j]$：前 $i$ 个村庄放置 $j$ 个邮局的最小距离和。

$w(l,r)$： 编号为 $[l,r]$ 之间的村庄中放一个邮局的最小距离和。

dp转移方程如下

​						$f[i][j] = min_{0≤k<i}f[k][j - 1] + w(k + 1,i)$

易知当区间内村庄数量为奇数时邮局放在最中间的村庄上是最优，为偶数时放在中间两个任一都是最优，于是 $w$ 有dp方程

​						$w[l][r] = w[l][r-1]+x[r]-x[⌊\frac{l+r}2⌋]$

$w$ 是满足四边形不等式的，且显然满足包含单调，于是 $f$ 也满足四边形不等式，根据定理3 $f[i][j]$ 的最优决策点 $p[i][j]$ 满足 

​						$p[i][j-1]≤p[i][j]≤p[i+1][j]$

此时我们需要处理最优决策点范围就大大缩小了，因为决策点需要用到 $p[i+1]$ 我们倒序求解。

```C++
// P4767 [IOI2000] 邮局 https://www.luogu.com.cn/problem/P4767
#include <bits/stdc++.h>
using namespace std;

const int N = 3010, inf = 0x3f3f3f3f;

int n, m, x[N], f[N][N], w[N][N], p[N][N];
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> x[i];
    sort(x + 1, x + 1 + n);

    for(int l = 1; l <= n; l ++){ // 预处理wij
        w[l][l] = 0;
        for(int r = l + 1; r <= n; r ++){
            w[l][r] = w[l][r - 1] + x[r] - x[(l + r) >> 1];
        }
    }

    memset(f, 0x3f, sizeof f);
    f[0][0] = 0;
    for(int j = 1; j <= m; j ++){
        p[n + 1][j] = n;
        for(int i = n; i >= 1; i --){
            for(int k = p[i][j - 1]; k <= p[i + 1][j]; k ++){
                if(f[i][j] > f[k][j - 1] + w[k + 1][i]){
                    f[i][j] = f[k][j - 1] + w[k + 1][i];
                    p[i][j] = k;
                }
            }
        }
    }

    cout << f[n][m];
    return 0;
}
```



# 字符串



## Hash

​		字符串hash：一般将字符串看做一个 $p$ 进制数，且各位数非 $0$ 例如小写字母可以赋值为 $a = 1,b = 2\dots,z=26$，一般 $p$ 取值为 $131$ 或 $13331$.

hash常用模数：

1. 用 unsigned long long 自然溢出取模。
2. int 范围内自定义质数 $1000000933$.
3. long long 范围内自定义质数 $100000000699$.

使用两种模数的双hash一般来说更能保证不会出现hash冲突。

```C++
#include <iostream>
#include <algorithm>
using namespace std;
#define ll long long
#define ull unsigned long long
const int mod = 1000000933;
const int base1 = 13331, base2 = 131;
const int N = 1e6 + 10, MAX = 1000000;
char a[N];
ull p1[N], h1[N];
ll p2[N], h2[N], n;
void init(){
    p1[0] = p2[0] = 1;
    for(int i = 1; i <= MAX; i ++){
        p1[i] = p1[i - 1] * base1;
        p2[i] = p2[i - 1] * base2 % mod;
    }
}

void Hash(){
    for(int i = 1; i <= n; i ++){
        h1[i] = h1[i - 1] * base1 + a[i];
        h2[i] = (h2[i - 1] * base2 % mod + a[i]) % mod;
    }
}

ull get_hash1(int l, int r){
    return h1[r] - h1[l - 1] * p1[r - l + 1];
}

ll get_hash2(int l, int r){
    return (h2[r] - h2[l - 1] * p2[r - l + 1] % mod + mod) % mod;
}
```



## KMP

$O(n)$ 求模式串 $t$ 在 文本串 $s$ 中出现的位置。

```C++
/*
fail[i]:指向不包括自己的最长后缀 与最长前缀匹配的位置
j代表的是前一个字符即以第i - 1个字符结尾的后缀 相匹配的最大前缀的(长度/下标)
1.若p[i] = p[j + 1]那么对于当前第i个字符可以继承第i-1个字符的匹配的基础上再+1
2.p[i] != p[j + 1]则需要回溯 回溯到以p[i]结尾的后缀能与某一前缀匹配
*/
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 1e5 + 10, M = 1e6 + 10;
char s[M],p[N];
int nex[N];
int main(){
    int n,m;
    scanf("%d%s%d%s",&n,p+1,&m,s+1);
    for(int i = 2, j = 0; i <= n; i ++)
    {
        while(j && p[i] != p[j + 1])j = nex[j];
        if(p[i] == p[j + 1]) j ++;
        nex[i] = j;         
    }
    
    for(int i = 1, j = 0; i <= m; i ++)
    {
        while(j && s[i] != p[j + 1])j = nex[j];
        if(s[i] == p[j + 1]) j ++;
        if(j == n){
            printf("%d ",i - n);
            j = nex[j];
        }
    }
    return 0;
}
```



## Manacher
线性时间复杂度求字符串中最大回文串长度。

预处理：
		manacher算法只能处理奇数长度的回文串，所以需要在每个字符之间插入一个不会影响结果的无关字符 以此将所有回文串扩展至奇数长度。一般回文串认为只包含英文字母，于是将 `$` 作为起始字符，在原字符串的开头结尾和两个字符之间插入一个 `#`，最后再插入 `^` 收尾，以此防止越界。设初始字符串为 $a_1a_2a_3……a_n$ 那么初始化后字符串为 ` $#a_1#a_2#……#a_n#^`.

原理：
		利用回文串的性质，维护当前出现过的回文串中右边界最靠右的回文串 维护其回文中心 $mid$ 和 右边界 $r_{max}$.

对于当前枚举到的回文中心 $i$ 分情况讨论

1. $i < r_{max}$ 即 $l < mid < i < r_{max}$ ，在 $[l, mid]$ 之间存在一个与 $i$ 对称的位置 $j$ （$mid * 2 - i$）
  - 若 $j$ 的回文左边界没有超过 $l$  那么 $p_i = p_j$。
  - 若 $j$ 的回文左边界超过 $l$，$i$ 的回文边界最多到 $r_{max}$， $p_i = r_{max} - i$，若是 $i$ 的回文边界会超过 $r_{max}$ 那么当前回文中心 $mid$ 的最右边界也会 $> r_{max}$ 不符合。
- 若 $j$ 左边界与 $mid$ 重合 此时需要循环去扩展当前回文最大右边界 $r_{max}$ 每次执行循环都是扩展右边界，最多到 $n$ 因此是线性的。


2. 若 $i >= r_{max}$ 更新 $mid$ 与 $r_{max}$ 即可。

```C++
#include <bits/stdc++.h>
using namespace std;

const int N = 1e7 + 1e6 + 100;

int n, p[N * 2]; // p[i]:以i为中心的回文串的最大回文半径
char a[N], s[N * 2]; // a:原串 s:预处理后的串

void manacher(){
    int k = 0;
    s[++ k] = '$'; s[++ k] = '#'; // 开始符号
    for(int i = 1; i <= n; i ++){
        s[++ k] = a[i]; s[++ k] = '#';
    }
    s[++ k] = '^'; // 截止符防止越界
    n = k;

    int mid = 0, max_r = 0; // mid:当前出现过的回文串中右边界最大的回文中心  max_r:出现过的回文最大右边界（开区间）
    for(int i = 1; i <= n; i ++){
        if(i < max_r) p[i] = min(p[mid * 2 - i], max_r - i); 
        else p[i] = 1;
        while(s[i - p[i]] == s[i + p[i]]) p[i] ++;
        if(i + p[i] > max_r){
            mid = i;
            max_r = i + p[i];
        }
    }
}

int main(){
    scanf("%s", a + 1);
    n = strlen(a + 1);

    manacher();

    int res = 0;
    for(int i = 1; i <= n; i ++) res = max(res, p[i]);
    cout << res - 1 << "\n"; // 原回文串长度 = 预处理后回文串的回文半径 - 1
    return 0;
}
```



## 字典树

一般空间大小为 $O(\sum_1^n \vert s_i\vert)$，其中 $\vert s \vert$ 为字符串长度。

$son[i][j]$：字符串前缀地址为 $i$，下一个字符为 $j$ 的地址。 

```C++
#include <string.h>
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 1e5 + 10;
int son[N][26],cnt[N],tot;//一般空间开到字符串总长度
void insert(char str[]){
    int p = 0;
    for(int i = 1; str[i]; i ++){
        int u = str[i] - 'a';
        if(!son[p][u]) son[p][u] = ++ tot;
        p = son[p][u];
    }
    cnt[p] ++;
}
void insert(int x) { // 0/1字典树 son[N * 30][2]
    int p = 0;
    for(int i = 30; i >= 0; i --){
        int u = x >> i & 1;
        if(!son[p][u]) son[p][u] = ++ tot;
        p = son[p][u];
    }
}
int query(char str[]){
    int p = 0;
    for(int i = 1; str[i]; i ++){
        int u = str[i] - 'a';
        if(!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
char str[N];
int main()
{
    int n;
    scanf("%d",&n);
    while(n --){
        char op[2];
        scanf("%s%s",op,str+1);
        if(*op == 'I')insert(str);
        else printf("%d\n",query(str));
    }
    return 0;
}
```



## AC自动机

```C++
//fail[i] : 在树中，以i结尾的后缀(不包括自己本身)能匹配的最长前缀的位置
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 1e6 + 10;
int tr[N][26],cnt[N],nex[N],idx;
char str[N];
void insert(char s[]){
    int p = 0;
    for(int i = 0; s[i]; i ++){
        int u = s[i] - 'a';
        if(!tr[p][u]) tr[p][u] = ++ idx;
        p = tr[p][u];
    }
    cnt[p] ++;
}
int q[N];
void build()
{
    int now = 1, tot = 0;
    for(int i = 0; i < 26; i ++){
        if(tr[0][i])q[++ tot] = tr[0][i];
    }
    while(now <= tot){
        int u = q[now ++];
        for(int i = 0; i < 26; i ++){
            int p = tr[u][i];
            if(!p) tr[u][i] = tr[nex[u]][i];
            else{
                nex[p] = tr[nex[u]][i];
                q[++ tot] = p;
            }
        }
    }
}
int main()
{
    int n;
    scanf("%d",&n);
    while(n --){
        scanf("%s",str);
        insert(str);
    }
    build();

    scanf("%s",str);
    int ans = 0;
    for(int i = 0,t = 0; str[i]; i ++){
        int u = str[i] - 'a';
        t = tr[t][u];
        int p = t;
        while(p && cnt[p] != -1)
        {
            ans += cnt[p];
            cnt[p] = -1;
            p = nex[p];
        }
    }
    printf("%d\n",ans);
    return 0;
}
```



## 回文自动机

```C++
//https://www.luogu.com.cn/problem/P5496
#include <string.h>
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 5e5 + 10;
char s[N];
int len[N],cnt[N],fail[N],tr[N][26],tot = 1;
//fail[i]:i号节点代表的最长回文后缀在回文树中的位置(不包括自己本身)
int getfail(int x,int i)
{
    while(i - len[x] - 1 <= 0 || s[i] != s[i - len[x] - 1])x = fail[x];//失配就去找下一个
    return x;//当之前存在回文串能和s[i]构成新的回文子串即返回
}
int main()
{
    scanf("%s",s + 1);
    int n = strlen(s + 1),last = 0;
    fail[0] = 1, len[0] = 0;//0号节点为偶根
    fail[1] = 0, len[1] = -1;//1号节点为奇根
    int cur = 0;//以第i个字符结尾的最长回文串在回文树中的位置
    for(int i=1;i<=n;i++)
    {
        s[i] = (s[i] - 97 + last) % 26;
        int pos = getfail(cur , i);//pos代表以s[i-1]结尾且能与新字符s[i]构成新的回文串的最长回文子串 在回文树中的位置
        if(!tr[pos][s[i]])//新回文串如果回文树中没有这个节点需要新建
        {
            fail[++ tot] = tr[getfail(fail[pos] , i)][s[i]];//求新回文串的fail指针
            tr[pos][s[i]] = tot;
            len[tot] = len[pos] + 2;
            cnt[tot] = cnt[fail[tot]] + 1;
        }
        cur = tr[pos][s[i]];
        last = cnt[cur];
        printf("%d ",last);
    }
    return 0;
}
```



# 博弈论

## 博弈树

```C++
/*
[蓝桥杯 2017 国 A] 填字母游戏 https://www.luogu.com.cn/problem/P8658
给定只有L,O,*的字符串，每轮可以将一个*变成O/L，当出现LOL时该选手胜利，给定字符串问先手的输/赢/平?
*/
#include <map>
#include <iostream>
#include <algorithm>
using namespace std;

map<string, int>mp;
 
int dfs(string &s){ // 博弈树
    // 搜索的每一个分支都对上一手的玩家的胜负进行判断
	if(mp.count(s)) return mp[s];
	if(s.find("LOL") != string::npos) return -1; // 当前人已经输了
	if(s.find("*") == string::npos) return 0; // 平局

	int res = -1; // 置当前为 -1
    // 从下一层决策树返回的结果取相反，对手必胜我必输反之亦然
	for (int i = 0; i < s.size(); i ++) {
		if(s[i] != '*') continue ;
		
		s[i] = 'L';
		res = max(res, -1 * dfs(s)); // 因为每人都取最优，所以当有一种决策更优时一定选择该决策能赢就赢，否则就逼平
		s[i] = '*';
		if(res == 1) break; // 有必胜的走法就选必胜，然后直接退出
		
		s[i] = 'O';
		res = max(res, -1 * dfs(s));
		s[i] = '*';
		if(res == 1) break;
	}
	mp[s] = res;
	return res;
}

int main(){
	ios::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	int t;
	cin >> t;
	while(t --){
		string s;
		cin >> s;
		cout << dfs(s) << "\n";
	}
	return 0;
}

```



## SG函数

**1. 平等组合游戏**

- 两人轮流走步。
- 有一个状态集，而且通常是有限的。
- 有一个终止状态，到达终止状态后游戏结束。
- 游戏可以在有限的步数内结束。
- 规定好了哪些状态转移是合法的。
- 所有规定对于两人是一样的。



**2. N状态（必胜状态），P状态（必败状态）**

**我们定义两个状态之间的转换：**

- 所有的终止状态都为P状态。
- 对于任意的N状态，存在至少一条路径可以转移到P状态。
- 对于任意的P状态，只能转移到N状态。



1. 当有多个局面可以同时选择时，赢/输取决于异或，例如nim游戏中 $n$ 堆石子，例如多个起点的有向图博弈，这些局面是同时存在的。

2. 而当局面只能选其一进行下去时，则是 $sg = mex$，例如求单个起点的有向图博弈，当选择某一条路径后就不可能返回走其他路了，于是在这些不能同时存在的选择中取 $mex$.

多起点的有向无环图博弈

```C++
// memset(sg, -1, sizeof sg);
#include <bits/stdc++.h>
using namespace std;

const int N = 2010;

vector<int> g[N];

int sg[N]; // sg[u] = mex{sg[v1],sg[v2],……,sg[vm]} 其中v为u的所有后继节点的sg值，mex为最小不存在的自然数

int get_mex(vector<int> &mex){
    sort(mex.begin(), mex.end());
    int siz = mex.size(), Mex = 0;
    for(int i = 0; i < siz; i ++){
        if(mex[i] == Mex) Mex ++;
        else if(mex[i] > Mex) return Mex;
    }
    return Mex;
}

int dfs(int u){
    if(sg[u] != -1) return sg[u];
    vector<int> mex;
    for(auto v : g[u]){
        mex.push_back(dfs(v));
    }
    return sg[u] = get_mex(mex);
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int n, m, k;
    cin >> n >> m >> k;

    for(int i = 1; i <= m; i ++){ // 给定有向无环图
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
    }

    memset(sg, -1, sizeof sg);
    int ans = 0;
    for(int i = 1; i <= k; i ++) { // 给定k个起点
        int x; cin >> x;
        ans ^= dfs(x); // nim 游戏
    }

    if(ans) cout << "win";
    else cout << "lose";
    return 0;
}
```



2023杭电多校2A sg打表

给定长为 $n$ 的段，和 $k$

1. 消除长度等于 $k$ 的连续怪物序列并要求两边的怪物序列不为空

2. 消除长度小于等于 $k$ 的连续怪物序列(必须一次消灭一段，若一段数量 $> k$ 不能从中挑一段消灭)

```C++
int dfs(int p){
    if(sg[p][k] != -1) return sg[p][k];
    if(p <= k) return sg[p][k] = 1;
    sg[p][k] = 0;
    vector<int> mex;
    for(int i = 2; i + k - 1 < p; i ++){
        // 当操作1取走中间段后，两边剩余的段是同时存在的状态，于是结果为sg[l] ^ sg[r]
        mex.push_back(dfs(i - 1) ^ dfs(p - k - (i - 1))); 
    }
    // 选择不同的中间段，形成的剩余两段也不同，是无法同时存在的局面取mex
    return sg[p][k] = get_mex(mex); 
}
```



# 计算几何



## 两矩形求交集

转化为计算线段区间交的问题，相交面积 $=$ 相交的水平的线段长度 $*$ 竖直方向线段长度。
坐标以 左下角 与 右上角 形式给出

```C++
#include<iostream>
#include<algorithm>
using namespace std;
struct Rec{
    int x1,y1,x2,y2;
    void input(){ cin >> x1 >> y1 >> x2 >> y2; }
};
int overlap_area(Rec &a, Rec &b){//计算线段交
    int x = max(0, min(a.x2, b.x2) - max(a.x1, b.x1));
    int y = max(0, min(a.y2, b.y2) - max(a.y1, b.y1));
    return x * y;
}

int main(){
    Rec a,b,c;
    a.input();
    b.input();
    c.input();
    cout << overlap_area(a,c) + overlap_area(b,c);C++
    return 0;
}
```



## 多矩形求并集（扫描线线段树 + 离散化）

```C++
 /*
给出的是坐标，但我们需要计算的是线段，于是我们存入时需要将r点坐标-1, 在线段树中计算时将每个区间再将r + 1
为什么要后面加回来，直接这里不减不行吗，对于叶子节点来说，1 + 2 + 3 
实际代表[v[1], v[2]] + [v[2], v[3]] ,如果不减那么答案将会是[v[1],v[1]] + [v[2],v[2]] + [v[3],v[3]] = 0.
也就是说存进线段树的看似是点v[i] 实际是v[i] ~ v[i + 1]这一段
*/
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
#define ll long long
const int N = 1e5 + 10;
struct seg{
    int x, y1, y2; int k;
    bool operator <(const seg &A)const{ return x < A.x; }
}s[N * 2];
vector<int>v;
struct node{
    int l, r, cnt;
    ll len; 
}tr[N * 8];

int get(int y){//得到的是离散化对应的点
    return lower_bound(v.begin(), v.end(), y) - v.begin();
}

void build(int p, int l, int r){
    tr[p] = {l, r, 0, 0};
    if(l == r) return ;
    int mid = (l + r) >> 1;
    build(p << 1, l, mid);
    build(p << 1 | 1, mid + 1, r);
}

void pushup(int p){
    //计算的线段长度，v[r + 1] - v[l] 如果不这么写，在叶子节点l = r tr[r] - tr[l] = 0,与线段的定义不符合 
    if(tr[p].cnt) tr[p].len = v[tr[p].r + 1] - v[tr[p].l];
    else if(tr[p].l != tr[p].r) tr[p].len = tr[p << 1].len + tr[p << 1 | 1].len;
    else tr[p].len = 0;
}

void update(int p, int l, int r, int k){
    if(tr[p].l >= l && tr[p].r <= r){
        tr[p].cnt += k;
        pushup(p);
        return ;
    }
    int mid = (tr[p].l + tr[p].r) >> 1;
    if(l <= mid) update(p << 1, l, r, k);
    if(r > mid) update(p << 1 | 1, l, r, k);
    pushup(p);
}

int main(){
    int n;
    scanf("%d",&n);
    for(int i = 1, j = 0; i <= n; i ++){
        int x1, y1, x2, y2;
        scanf("%d%d%d%d",&x1, &y1, &x2, &y2);//给出坐标是左上角和右下角
        s[++ j] = {x1, y1, y2, 1};
        s[++ j] = {x2, y1, y2, -1};
        v.push_back(y1); v.push_back(y2);
    }
    sort(s + 1, s + 1 + 2 * n);
    sort(v.begin(), v.end());
    unique(v.begin(), v.end());

    build(1, 0, v.size());
    
    ll ans = 0;
    for(int i = 1; i <= 2 * n; i ++){
        if(i > 1) ans += 1ll * (s[i].x - s[i - 1].x) * tr[1].len;
        update(1, get(s[i].y1), get(s[i].y2) - 1, s[i].k);
    }
    printf("%lld",ans);
    return 0;
}
```



## kuangbin二维（点线圆）

```C++
#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-8;
const double inf = 1e20;
const double pi = acos(-1.0);
const int maxp = 1010;
//`Compares a double to zero`
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
//square of a double
inline double sqr(double x){return x*x;}
struct Point{
	double x,y;
	Point(){}
	Point(double _x,double _y){
		x = _x;
		y = _y;
	}
	void input(){
		scanf("%lf%lf",&x,&y);
	}
	void output(){
		printf("%.2f %.2f\n",x,y);
	}
	bool operator == (Point b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0;
	}
	bool operator < (Point b)const{
		return sgn(x-b.x)== 0?sgn(y-b.y)<0:x<b.x;
	}
	Point operator -(const Point &b)const{
		return Point(x-b.x,y-b.y);
	}
	//叉积
	double operator ^(const Point &b)const{
		return x*b.y - y*b.x;
	}
	//点积
	double operator *(const Point &b)const{
		return x*b.x + y*b.y;
	}
	//返回长度
	double len(){
		return hypot(x,y);//库函数
	}
	//返回长度的平方
	double len2(){
		return x*x + y*y;
	}
	//返回两点的距离
	double distance(Point p){
		return hypot(x-p.x,y-p.y);
	}
	Point operator +(const Point &b)const{
		return Point(x+b.x,y+b.y);
	}
	Point operator *(const double &k)const{
		return Point(x*k,y*k);
	}
	Point operator /(const double &k)const{
		return Point(x/k,y/k);
	}
	//`计算pa  和  pb 的夹角`
	//`就是求这个点看a,b 所成的夹角`
	//`测试 LightOJ1203`
	double rad(Point a,Point b){
		Point p = *this;
		return fabs(atan2( fabs((a-p)^(b-p)),(a-p)*(b-p) ));
	}
	//`化为长度为r的向量`
	Point trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point(x*r,y*r);
	}
	//`逆时针旋转90度`
	Point rotleft(){
		return Point(-y,x);
	}
	//`顺时针旋转90度`
	Point rotright(){
		return Point(y,-x);
	}
	//`绕着p点逆时针旋转angle弧度`
	Point rotate(Point p,double angle){
		Point v = (*this) - p;
		double c = cos(angle), s = sin(angle);
		return Point(p.x + v.x*c - v.y*s,p.y + v.x*s + v.y*c);
	}
	/*
    atan2(double y,double x) 返回的主要是y/x的反正切值，在区间[-pi, +pi]弧度
    double angle = atan2(t[i].y,t[i].x) - atan2(s[j].y,s[j].x);//从i点旋转到j点
    */
};
struct Line{
	Point s,e;
	Line(){}
	Line(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	bool operator ==(Line v) {
		return (s == v.s)&&(e == v.e);
	}
	//`根据一个点和倾斜角angle确定直线,0<=angle<pi`
	Line(Point p,double angle){
		s = p;
		if(sgn(angle-pi/2) == 0){
			e = (s + Point(0,1));
		}
		else{
			e = (s + Point(1,tan(angle)));
		}
	}
	//ax+by+c=0
	Line(double a,double b,double c){//ax+by+c=0
		if(!sgn(a)){
			s=Point(0,-c/b),e=Point(1,-c/b);
			if(sgn(b)>0)adjust();   // 保证点是逆时针 有时候需要变号
		}
		else if(!sgn(b)){
			s=Point(-c/a,0),e=Point(-c/a,1);
			if(sgn(a)<0)adjust();
		}
		else{
			s=Point(0,-c/b),e=Point(1,(-c-a)/b);
			if(sgn(b)>0)adjust();
		}
	}
	void input(){
		s.input();
		e.input();
	}
	void adjust(){
		if(e < s)swap(s,e);
	}
	//求线段长度
	double length(){
		return s.distance(e);
	}
	//`返回直线倾斜角 0<=angle<pi`
	double angle(){
		double k = atan2(e.y-s.y,e.x-s.x);
		if(sgn(k) < 0)k += pi;
		if(sgn(k-pi) == 0)k -= pi;
		return k;
	}
	//`点和直线关系`
	//`1  在左侧`
	//`2  在右侧`
	//`3  在直线上`
	int relation(Point p){
		int c = sgn((p-s)^(e-s));
		if(c < 0)return 1;
		else if(c > 0)return 2;
		else return 3;
	}
	// 点在线段上的判断
	bool pointonseg(Point p){
		return sgn((p-s)^(e-s)) == 0 && sgn((p-s)*(p-e)) <= 0;
	}
	//`两向量平行(对应直线平行或重合)`
	bool parallel(Line v){
		return sgn((e-s)^(v.e-v.s)) == 0;
	}
	//`两线段相交判断`
	//`2 规范相交`
	//`1 非规范相交`
	//`0 不相交`
	int segcrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		int d3 = sgn((v.e-v.s)^(s-v.s));
		int d4 = sgn((v.e-v.s)^(e-v.s));
		if( (d1^d2)==-2 && (d3^d4)==-2 )return 2;
		return (d1==0 && sgn((v.s-s)*(v.s-e))<=0) ||
			(d2==0 && sgn((v.e-s)*(v.e-e))<=0) ||
			(d3==0 && sgn((s-v.s)*(s-v.e))<=0) ||
			(d4==0 && sgn((e-v.s)*(e-v.e))<=0);
	}
	//`直线和线段相交判断`
	//`-*this line   -v seg`
	//`2 规范相交`
	//`1 非规范相交`
	//`0 不相交`
	int linecrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		if((d1^d2)==-2) return 2;
		return (d1==0||d2==0);
	}
	//`两直线关系`
	//`0 平行`
	//`1 重合`
	//`2 相交`
	int linecrossline(Line v){
		if((*this).parallel(v))
			return v.relation(s)==3;
		return 2;
	}
	//`求两直线的交点`
	//`要保证两直线不平行或重合`
	Point crosspoint(Line v){
		double a1 = (v.e-v.s)^(s-v.s);
		double a2 = (v.e-v.s)^(e-v.s);
		return Point((s.x*a2-e.x*a1)/(a2-a1),(s.y*a2-e.y*a1)/(a2-a1));
	}
	//点到直线的距离
	double dispointtoline(Point p){
		return fabs((p-s)^(e-s))/length();
	}
	//点到线段的距离
	double dispointtoseg(Point p){
		if(sgn((p-s)*(e-s))<0 || sgn((p-e)*(s-e))<0)
			return min(p.distance(s),p.distance(e));
		return dispointtoline(p);
	}
	//`返回线段到线段的距离`
	//`前提是两线段不相交，相交距离就是0了`
	double dissegtoseg(Line v){
		return min(min(dispointtoseg(v.s),dispointtoseg(v.e)),min(v.dispointtoseg(s),v.dispointtoseg(e)));
	}
	//`返回点p在直线上的投影`
	Point lineprog(Point p){
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`返回点p关于直线的对称点`
	Point symmetrypoint(Point p){
		Point q = lineprog(p);
		return Point(2*q.x-p.x,2*q.y-p.y);
	}
};

// 圆相关
struct circle{
	Point p;//圆心
	double r;//半径
	circle(){}
	circle(Point _p,double _r){
		p = _p;
		r = _r;
	}
	circle(double x,double y,double _r){
		p = Point(x,y);
		r = _r;
	}
	//`三角形的外接圆`
	//`需要Point的+ /  rotate()  以及Line的crosspoint()`
	//`利用两条边的中垂线得到圆心`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c){
		Line u = Line((a+b)/2,((a+b)/2)+((b-a).rotleft()));
		Line v = Line((b+c)/2,((b+c)/2)+((c-b).rotleft()));
		p = u.crosspoint(v);
		r = p.distance(a);
	}
	//`三角形的内切圆`
	//`参数bool t没有作用，只是为了和上面外接圆函数区别`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c,bool t){
		Line u,v;
		double m = atan2(b.y-a.y,b.x-a.x), n = atan2(c.y-a.y,c.x-a.x);
		u.s = a;
		u.e = u.s + Point(cos((n+m)/2),sin((n+m)/2));
		v.s = b;
		m = atan2(a.y-b.y,a.x-b.x) , n = atan2(c.y-b.y,c.x-b.x);
		v.e = v.s + Point(cos((n+m)/2),sin((n+m)/2));
		p = u.crosspoint(v);
		r = Line(a,b).dispointtoseg(p);
	}
	//输入
	void input(){
		p.input();
		scanf("%lf",&r);
	}
	//输出
	void output(){
		printf("%.2lf %.2lf %.2lf\n",p.x,p.y,r);
	}
	bool operator == (circle v){
		return (p==v.p) && sgn(r-v.r)==0;
	}
	bool operator < (circle v)const{
		return ((p<v.p)||((p==v.p)&&sgn(r-v.r)<0));
	}
	//面积
	double area(){
		return pi*r*r;
	}
	//周长
	double circumference(){
		return 2*pi*r;
	}
	//`点和圆的关系`
	//`0 圆外`
	//`1 圆上`
	//`2 圆内`
	int relation(Point b){
		double dst = b.distance(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r)==0)return 1;
		return 0;
	}
	//`线段和圆的关系`
	//`比较的是圆心到线段的距离和半径的关系`
	int relationseg(Line v){
		double dst = v.dispointtoseg(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`直线和圆的关系`
	//`比较的是圆心到直线的距离和半径的关系`
	int relationline(Line v){
		double dst = v.dispointtoline(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`两圆的关系`
	//`5 相离`
	//`4 外切`
	//`3 相交`
	//`2 内切`
	//`1 内含`
	//`需要Point的distance`
	//`测试：UVA12304`
	int relationcircle(circle v){
		double d = p.distance(v.p);
		if(sgn(d-r-v.r) > 0)return 5;
		if(sgn(d-r-v.r) == 0)return 4;
		double l = fabs(r-v.r);
		if(sgn(d-r-v.r)<0 && sgn(d-l)>0)return 3;
		if(sgn(d-l)==0)return 2;
		if(sgn(d-l)<0)return 1;
	}
	//`求两个圆的交点，返回0表示没有交点，返回1是一个交点，2是两个交点`
	//`需要relationcircle`
	//`测试：UVA12304`
	int pointcrosscircle(circle v,Point &p1,Point &p2){
		int rel = relationcircle(v);
		if(rel == 1 || rel == 5)return 0;
		double d = p.distance(v.p);
		double l = (d*d+r*r-v.r*v.r)/(2*d);
		double h = sqrt(r*r-l*l);
		Point tmp = p + (v.p-p).trunc(l);
		p1 = tmp + ((v.p-p).rotleft().trunc(h));
		p2 = tmp + ((v.p-p).rotright().trunc(h));
		if(rel == 2 || rel == 4)
			return 1;
		return 2;
	}
	//`求直线和圆的交点，返回交点个数`
	int pointcrossline(Line v,Point &p1,Point &p2){
		if(!(*this).relationline(v))return 0;
		Point a = v.lineprog(p);
		double d = v.dispointtoline(p);
		d = sqrt(r*r-d*d);
		if(sgn(d) == 0){
			p1 = a;
			p2 = a;
			return 1;
		}
		p1 = a + (v.e-v.s).trunc(d);
		p2 = a - (v.e-v.s).trunc(d);
		return 2;
	}
	//`得到过a,b两点，半径为r1的两个圆`
	int gercircle(Point a,Point b,double r1,circle &c1,circle &c2){
		circle x(a,r1),y(b,r1);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r;
		return t;
	}
	//`得到与直线u相切，过点q,半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Point q,double r1,circle &c1,circle &c2){
		double dis = u.dispointtoline(q);
		if(sgn(dis-r1*2)>0)return 0;
		if(sgn(dis) == 0){
			c1.p = q + ((u.e-u.s).rotleft().trunc(r1));
			c2.p = q + ((u.e-u.s).rotright().trunc(r1));
			c1.r = c2.r = r1;
			return 2;
		}
		Line u1 = Line((u.s + (u.e-u.s).rotleft().trunc(r1)),(u.e + (u.e-u.s).rotleft().trunc(r1)));
		Line u2 = Line((u.s + (u.e-u.s).rotright().trunc(r1)),(u.e + (u.e-u.s).rotright().trunc(r1)));
		circle cc = circle(q,r1);
		Point p1,p2;
		if(!cc.pointcrossline(u1,p1,p2))cc.pointcrossline(u2,p1,p2);
		c1 = circle(p1,r1);
		if(p1 == p2){
			c2 = c1;
			return 1;
		}
		c2 = circle(p2,r1);
		return 2;
	}
	//`同时与直线u,v相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Line v,double r1,circle &c1,circle &c2,circle &c3,circle &c4){
		if(u.parallel(v))return 0;//两直线平行
		Line u1 = Line(u.s + (u.e-u.s).rotleft().trunc(r1),u.e + (u.e-u.s).rotleft().trunc(r1));
		Line u2 = Line(u.s + (u.e-u.s).rotright().trunc(r1),u.e + (u.e-u.s).rotright().trunc(r1));
		Line v1 = Line(v.s + (v.e-v.s).rotleft().trunc(r1),v.e + (v.e-v.s).rotleft().trunc(r1));
		Line v2 = Line(v.s + (v.e-v.s).rotright().trunc(r1),v.e + (v.e-v.s).rotright().trunc(r1));
		c1.r = c2.r = c3.r = c4.r = r1;
		c1.p = u1.crosspoint(v1);
		c2.p = u1.crosspoint(v2);
		c3.p = u2.crosspoint(v1);
		c4.p = u2.crosspoint(v2);
		return 4;
	}
	//`同时与不相交圆cx,cy相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(circle cx,circle cy,double r1,circle &c1,circle &c2){
		circle x(cx.p,r1+cx.r),y(cy.p,r1+cy.r);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r1;
		return t;
	}

	//`过一点作圆的切线(先判断点和圆的关系)`
	//`测试：UVA12304`
	int tangentline(Point q,Line &u,Line &v){
		int x = relation(q);
		if(x == 2)return 0;
		if(x == 1){
			u = Line(q,q + (q-p).rotleft());
			v = u;
			return 1;
		}
		double d = p.distance(q);
		double l = r*r/d;
		double h = sqrt(r*r-l*l);
		u = Line(q,p + ((q-p).trunc(l) + (q-p).rotleft().trunc(h)));
		v = Line(q,p + ((q-p).trunc(l) + (q-p).rotright().trunc(h)));
		return 2;
	}
	//`求两圆相交的面积`
	double areacircle(circle v){
		int rel = relationcircle(v);
		if(rel >= 4)return 0.0;
		if(rel <= 2)return min(area(),v.area());
		double d = p.distance(v.p);
		double hf = (r+v.r+d)/2.0;
		double ss = 2*sqrt(hf*(hf-r)*(hf-v.r)*(hf-d));
		double a1 = acos((r*r+d*d-v.r*v.r)/(2.0*r*d));
		a1 = a1*r*r;
		double a2 = acos((v.r*v.r+d*d-r*r)/(2.0*v.r*d));
		a2 = a2*v.r*v.r;
		return a1+a2-ss;
	}
	//`求圆和三角形pab的相交面积`
	//`测试：POJ3675 HDU3982 HDU2892`
	double areatriangle(Point a,Point b){
		if(sgn((p-a)^(p-b)) == 0)return 0.0;
		Point q[5];
		int len = 0;
		q[len++] = a;
		Line l(a,b);
		Point p1,p2;
		if(pointcrossline(l,q[1],q[2])==2){
			if(sgn((a-q[1])*(b-q[1]))<0)q[len++] = q[1];
			if(sgn((a-q[2])*(b-q[2]))<0)q[len++] = q[2];
		}
		q[len++] = b;
		if(len == 4 && sgn((q[0]-q[1])*(q[2]-q[1]))>0)swap(q[1],q[2]);
		double res = 0;
		for(int i = 0;i < len-1;i++){
			if(relation(q[i])==0||relation(q[i+1])==0){
				double arg = p.rad(q[i],q[i+1]);
				res += r*r*arg/2.0;
			}
			else{
				res += fabs((q[i]-p)^(q[i+1]-p))/2.0;
			}
		}
		return res;
	}
};
```



## kuangbin三维（点线面）

```C++
#include <bits/stdc++.h>
using namespace std;
const double pi = acos(-1.0);//π的较精确值
const double eps = 1e-8;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0,double _y = 0,double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
  double rand_eps() {
		return ((double)rand() / RAND_MAX - 0.5) * eps;
	}
	void shake() {
		x += rand_eps();
		y += rand_eps();
		z += rand_eps();
	}
	void output(){
		printf("%.8lf %.8lf %.8lf\n",x,y,z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	bool operator <(const Point3 &b)const{
		return sgn(x-b.x)==0?(sgn(y-b.y)==0?sgn(z-b.z)<0:y<b.y):x<b.x;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//点乘
	double operator *(const Point3 &b)const{
		return x*b.x+y*b.y+z*b.z;
	}
	//叉乘
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
	double rad(Point3 a,Point3 b){
		Point3 p = (*this);
		return acos( ( (a-p)*(b-p) )/ (a.distance(p)*b.distance(p)) );
	}
	//变换长度
	Point3 trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point3(x*r,y*r,z*r);
	}
};
struct Line3
{
	Point3 s,e;
	Line3(){}
	Line3(Point3 _s,Point3 _e)
	{
		s = _s;
		e = _e;
	}
	bool operator ==(const Line3 v)
	{
		return (s==v.s)&&(e==v.e);
	}
	void input()
	{
		s.input();
		e.input();
	}
	double length()
	{
		return s.distance(e);
	}
	//点到直线距离
	double dispointtoline(Point3 p)
	{
		return ((e-s)^(p-s)).len()/s.distance(e);
	}
	//点到线段距离
	double dispointtoseg(Point3 p)
	{
		if(sgn((p-s)*(e-s)) < 0 || sgn((p-e)*(s-e)) < 0)
			return min(p.distance(s),e.distance(p));
		return dispointtoline(p);
	}
	//`返回点p在直线上的投影`
	Point3 lineprog(Point3 p)
	{
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`p绕此向量逆时针arg弧度`
	Point3 rotate(Point3 p,double ang)
	{
		if(sgn(((s-p)^(e-p)).len()) == 0)return p;
		Point3 f1 = (e-s)^(p-s);
		Point3 f2 = (e-s)^(f1);
		double len = ((s-p)^(e-p)).len()/s.distance(e);
		f1 = f1.trunc(len); f2 = f2.trunc(len);
		Point3 h = p+f2;
		Point3 pp = h+f1;
		return h + ((p-h)*cos(ang)) + ((pp-h)*sin(ang));
	}
	//`点在直线上`
	bool pointonseg(Point3 p)
	{
		return sgn( ((s-p)^(e-p)).len() ) == 0 && sgn((s-p)*(e-p)) == 0;
	}
};
struct Plane
{
	Point3 a,b,c,o;//`平面上的三个点，以及法向量`
	Plane(){}
	Plane(Point3 _a,Point3 _b,Point3 _c)
	{
		a = _a;
		b = _b;
		c = _c;
		o = pvec();
	}
	Point3 pvec()
	{
		return (b-a)^(c-a);
	}
	//`ax+by+cz+d = 0`
	Plane(double _a,double _b,double _c,double _d)
	{
		o = Point3(_a,_b,_c);
		if(sgn(_a) != 0)
			a = Point3((-_d-_c-_b)/_a,1,1);
		else if(sgn(_b) != 0)
			a = Point3(1,(-_d-_c-_a)/_b,1);
		else if(sgn(_c) != 0)
			a = Point3(1,1,(-_d-_a-_b)/_c);
	}
	//`点在平面上的判断`
	bool pointonplane(Point3 p)
	{
		return sgn((p-a)*o) == 0;
	}
	//`两平面夹角`
	double angleplane(Plane f)
	{
		return acos(o*f.o)/(o.len()*f.o.len());
	}
	//`平面和直线的交点，返回值是交点个数`
	int crossline(Line3 u,Point3 &p)
	{
		double x = o*(u.e-a);
		double y = o*(u.s-a);
		double d = x-y;
		if(sgn(d) == 0)return 0;
		p = ((u.s*x)-(u.e*y))/d;
		return 1;
	}
	//`点到平面最近点(也就是投影)`
	Point3 pointtoplane(Point3 p)
	{
		Line3 u = Line3(p,p+o);
		crossline(u,p);
		return p;
	}
	//`平面和平面的交线`
	int crossplane(Plane f,Line3 &u)
	{
		Point3 oo = o^f.o;
		Point3 v = o^oo;
		double d = fabs(f.o*v);
		if(sgn(d) == 0)return 0;
		Point3 q = a + (v*(f.o*(f.a-a))/d);
		u = Line3(q,q+oo);
		return 1;
	}
};
```



## 最小覆盖问题

### 最小圆覆盖（几何）

最小圆覆盖问题：给定 $n$ 个点的平面坐标，求一个半径最小的圆，把 $n$ 个点全部包围，部分点在圆上。

（1）加第 $1$ 个点 $P_1$。$C_1$ 的圆心就是 $P_1$，半径为 $0$。

（2）加第 $2$ 个点 $P_2$。新的 $C_2$ 的圆心是线段 $P_1,P_2$ 的中心，半径为两点距离的一半。这一步操作是两点定圆。

（3）加第 $3$ 个点 $P_3$。若 $P_3$ 在圆内或圆上，忽略；若不在，则以 $P_3$ 为圆心，重复（1）和（2），若还是不行则		  用三点定圆。
（4）加第 $4$ 个点 $P_4$。若 $P_4$ 在圆内或圆上，忽略；若不在，则以 $P_4$ 为圆心，从前三个点中选择一个点重复（1）和（2）即两点定圆，若还是不行则选三个点进行三点定圆(一定有)。

（5）继续加入新的点。

复杂度分析：3层for循环，貌似是 $O(n^3)$，但是当点的分布是随机的时候，可以通过概论计算得到实际复杂度接近 $O(n)$，代码中使用random_shuffle()函数实现。

```C++
#include <bits/stdc++.h>
using namespace std;

const double eps = 1e-8;
const int maxn = 1e5 + 10;

int sgn(double x) {
    if (fabs(x)<eps) return 0;
    else return x<0? -1:1;
}
struct Point{
    double x, y;
};

double Distance(Point A, Point B){
    return hypot(A.x-B.x,A.y-B.y);
}

Point circle_center(const Point a, const Point b, const Point c)//求三角形abc的外接圆圆心
{
    Point center;
    double a1 = b.x-a.x, b1 = b.y-a.y, c1 = (a1*a1+b1*b1)/2;
    double a2 = c.x-a.x, b2 = c.y-a.y, c2 = (a2*a2+b2*b2)/2;
    double d = a1*b2-a2*b1;
    center.x = a.x+(c1*b2-c2*b1)/d;
    center.y = a.y+(a1*c2-a2*c1)/d;
    return center;
}

void min_cover_circle(Point *p, int n,Point &c, double &r){ //求最小圆覆盖，返回圆心c和半径r:
    random_shuffle(p,p+n); //打乱所有点
    c = p[0]; r = 0; //第一个点
    for (int i = 1; i < n; i ++){ //剩下所有点
        if (sgn(Distance(p[i],c)-r) > 0){ //pi在圆外部
            c = p[i]; r = 0; //将圆心设为pi半径为0
            for (int j = 0; j < i; ++j){ //重新检查前面的点
                if(sgn(Distance(p[j],c)-r) > 0){ //两点定圆
                    c.x = (p[i].x+p[j].x)/2;
                    c.y = (p[i].y+p[j].y)/2;
                    r = Distance(p[j],c);
                    for (int k = 0; k < j; ++k){
                        if(sgn(Distance(p[k],c)-r) > 0){
                            c = circle_center(p[i],p[j],p[k]);
                            r = Distance(p[i],c);
                        }
                    }
                }
            }
        }
    }
}

int main(){
    int n;
    Point p[maxn];
    Point c;
    double r;
    while(~scanf("%d",&n) && n){
        for(int i = 0; i < n; i ++){
            scanf("%lf%lf",&p[i].x,&p[i].y);
        }
        min_cover_circle(p,n,c,r);
        printf("%.10lf\n%.10lf %.10lf\n",r,c.x,c.y);
    }
    return 0;
}
```



### 最小球覆盖（模拟退火）

```C++
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 105;
const double eps = 1e-8;
struct Point{
    double x,y,z;
    Point(double _x = 0, double _y = 0, double _z = 0){
        x = _x; y = _y; z = _z;
    }
};
Point Dots[MAXN];
int n;

double Distance(Point a,Point b){
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}
double get_rte(){
    double delta = 100000; // 温度，温度越大越精确， 时间复杂度越高 
    double ret = 1e9, mt;
    Point z = Point();
    int s = 0;
    while(delta > eps){
        for(int i = 1; i <= n; i ++){
            if(Distance(z, Dots[s]) < Distance(z, Dots[i])) s = i;
        }
        mt = Distance(z,Dots[s]);
        ret = min(ret, mt);
        z.x += (Dots[s].x - z.x) / mt * delta;
        z.y += (Dots[s].y - z.y) / mt * delta;
        z.z += (Dots[s].z - z.z) / mt * delta;
        delta *= 0.98;
    }
    return ret;
}

int main(){
	scanf("%d",&n);
    for(int i = 1; i <= n; i ++){
        double x,y,z;
        scanf("%lf%lf%lf",&x,&y,&z);
        Dots[i] = Point(x,y,z);
    }
    printf("%.10lf",get_rte());
    return 0;
}
```

