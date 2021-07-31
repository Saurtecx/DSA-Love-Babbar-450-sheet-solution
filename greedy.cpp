
// Greedy

//1>> N meetings in one room // TC O(NLogN)
struct Meeting {
    int id,start,finish;
};
bool comparision(Meeting a, Meeting b){
    if (a.finish == b.finish) return (a.id< b.id);
    return (a.finish < b.finish);
}
int maxMeetings(int start[], int end[], int n){
    Meeting arr[n];
    for(int i=0; i<n; i++){
        arr[i].id = i;
        arr[i].start = start[i];
        arr[i].finish = end[i];
    }
    sort(arr,arr+n,comparision);
    int j = 0;
    int count = 1;
    for(int i=1; i<n; i++){
        if(arr[j].finish < arr[i].start){
            j = i;
            count++;
        }
    }
    return count;
}

//2>> Job Sequencing Problem // TC o(nn) // SC o(n)
bool mycompare(Job a, Job b){
return a.profit > b.profit;
}
pair<int,int> JobScheduling(Job arr[], int n){
  unordered_map<int, int> mp;
  int profit=0, c=0;
  sort(arr, arr+n, mycompare);
for(int i=0; i<n; i++){
    for(int j=arr[i].dead; j>0; j--){
       if(mp.count(j) == 0){
           mp[j]++;
           profit+=arr[i].profit;
           c++;
           break;
        }
    }
 }
 return {c, profit};
}

//3>> Huffman Coding // TC o(nlogn)
struct Node{
    int data;
    Node*left;
    Node*right;
    Node(int x){
        data=x;
        left=NULL;
        right=NULL;
    }
};
struct cmp{
  bool operator()( Node*a, Node*b) {
      return a->data>b->data;
  }
};
void print(Node*root,string s){
    if(root->left==NULL && root->right==NULL){
        cout<<s<<" ";
        return;
    }
    print(root->left,s+"0");
    print(root->right,s+"1");
}
int main(){
    int t,n;
    cin>>t;
    string str;
    while(t--){
        cin>>str;
        n=str.length();int arr[n];
        priority_queue<Node*,vector<Node*>,cmp>pq;
        for(int i=0;i<n;i++){
            cin>>arr[i];
            Node*newnode=new Node(arr[i]);
            pq.push(newnode);
        }
        while(pq.size()!=1){
            Node* m1=pq.top();
            pq.pop();
            Node* m2=pq.top();
            pq.pop();
            Node* temp=new Node(m1->data+m2->data);
            temp->left=m1;
            temp->right=m2;
            pq.push(temp);
        }
       Node* root=pq.top();
        string str="";
        print(root,str);
        cout<<endl;
    }
}

//4>> Water connection problem // TC o(n) //
void dfs(vector<int> visited, vector<int> pipe_start[], vector<int> dia,
         int start, int &min_dia, int &last_stop){
    visited[start] = 1;
    for(auto j=pipe_start[start].begin(); j != pipe_start[start].end(); j++){
        min_dia = min(min(dia[start], min_dia), dia[*j]);
        last_stop = *j;
        dfs(visited, pipe_start, dia, *j, min_dia, last_stop);
    }
}

int main(){
	int tests;
	cin >> tests;
	while(tests--){
	    int houses, pipes;
	    cin >> houses >> pipes;
	    vector<int> pipe_start[houses+1];
	    vector<int> adj[houses+1];
	    vector<int> dia(houses+1, 101);
	    for(int i=0; i<pipes; i++){
	        int start,end,diameter;
	        cin >> start >> end >> diameter;
	        //Setting diameter
	        dia[start] = diameter;
	        //Storing pipe starts
	        pipe_start[start].push_back(end);
	        //Making the adjacency list (unordered)
	        adj[start].push_back(end);
	        adj[end].push_back(start);
	    }
	    int count = 0;   //pair of tank and tap
	    int result[100][3];
	    vector<int> visited(houses+1, 0);
	    for(int i=1; i<=houses; i++){
	        if(visited[i] == 0 && pipe_start[i].size()>0 && adj[i].size()==1){
	            int min_dia = INT_MAX;
	            int last_stop;
	            dfs(visited, pipe_start, dia, i, min_dia, last_stop);
	            result[count][0] = i;
	            result[count][1] = last_stop;
	            result[count][2] = min_dia;
	            count++;
	        }
	    }
	    cout<<count<<endl;
	    for(int i=0;i<count; i++){
	        cout<<result[i][0]<<" ";
	        cout<<result[i][1]<<" ";
	        cout<<result[i][2]<<" "<<endl;
	    }
	}
	return 0;
}

//5>> Fractional Knapsack Problem // TC o(nlogn)
bool cmp(struct Item a, struct Item b){
 double r1 = (double)a.value / a.weight;
 double r2 = (double)b.value / b.weight;
 return r1 > r2;
}
double fractionalKnapsack(int W, Item arr[], int n){
  sort(arr, arr + n, cmp);
  double finalvalue = 0.0;
  for (int i = 0; i < n; i++){
    if ( arr[i].weight <= W){
      W-= arr[i].weight;
      finalvalue += arr[i].value;
    }
    else{
      finalvalue += arr[i].value * ((double) W / arr[i].weight);
      break;
    }
  }
return finalvalue;
}

//6>> Greedy Algorithm to find Minimum number of Coins
// TC o(nlogn) // SC o(n)
signed main(){
	int t,n,k;  cin>>t;
	while(t--){
	    cin>> n >> k;
	    vector<int> a(n);
	    for(int i = 0 ; i < n ; i++)  cin>> a[i];
	    sort(a.begin(),a.end());
	    vector<int> p(n);  p[0] = a[0];
	    for(int i = 1 ; i < n ; i++)
	    p[i] = p[i-1] + a[i];
	    int ans = INT_MAX , prev = 0;
	    for(int i = 0 ; i < n ; i++){
	        int pos = upper_bound(a.begin()+i,a.end(),a[i]+k) - a.begin();
	        if(i > 0 && a[i] != a[i-1])  prev = p[i-1];
	        ans = min(ans , prev + p[n-1] - p[pos-1] - (n-pos) * (a[i]+k));
	    }
	    cout<< ans << endl;
	}
	return 0;
}

//7>> Maximum trains for which stoppage can be provided // TC o(nn)
// n-> plateform // m-> train;
int maxStop(int arr[][3]){
    vector<pair<int, int> > vect[n + 1];
    for (int i = 0; i < m; i++)
        vect[arr[i][2]].push_back(
             make_pair(arr[i][1], arr[i][0]));
    for (int i = 0; i <= n; i++)
        sort(vect[i].begin(), vect[i].end());
    int count = 0;
    for (int i = 0; i <= n; i++) {
        if (vect[i].size() == 0) continue;
        int x = 0;
        count++;
        for (int j = 1; j < vect[i].size(); j++) {
            if (vect[i][j].second>=vect[i][x].first) {
                x = j;
                count++;
            }
        }
    }
    return count;
}

//8>> minimum number of platforms required for the railway station
// so that no train is kept waiting // TC o(nlogn)
int findPlatform(int arr[], int dep[], int n){
	sort(arr, arr+n);
	sort(dep, dep+n);
	int pf=1;
	int j=0;
	for(int i=1;i<n;i++){
	    if(arr[i]<=dep[j]) pf++;
	    else j++;
	}
	return pf;
}

//9>> Buy Maximum Stocks if i stocks can be bought on i-th day
// TC o(nlogn)
int buyMaximumProducts(int n, int k, int price[]) {
    vector<pair<int, int> > v;
    for (int i = 0; i < n; ++i)
        v.push_back(make_pair(price[i], i + 1));
    sort(v.begin(), v.end());
    int ans = 0;
    for (int i = 0; i < n; ++i) {
        ans += min(v[i].second, k / v[i].first);
        k -= v[i].first * min(v[i].second,(k / v[i].first));
    }
    return ans;
}

//10>> minimum amount of money you have to spend to buy all
// the N different candies. Secondly, you have to tell what is the
//maximum amount of money you have to spend to buy all the
// N different candies // TC o(nlogn)
int n,k,mincost=0,maxcost=0;
        cin>>n>>k;
        vector<int> v(n);
        for(int i=0;i<n;i++) cin>>v[i];
        sort(v.begin(),v.end());
        int i=0,j=n-1;
        while(i<=j){
            mincost+=v[i++];
            j-=k;
        }
        i=0,j=n-1;
        while(i<=j){
            maxcost+=v[j--];
            i+=k;
        }
        cout<<mincost<<" "<<maxcost<<endl;

//11>> Minimize Cash Flow among a given set of friends who have
// borrowed money from each other // TC o(nn)
int getMin(int arr[]){
    int minInd = 0;
    for(int i=1; i<N; i++)
        if (arr[i] < arr[minInd]) minInd = i;
    return minInd;
}
int getMax(int arr[]) {
    int maxInd = 0;
    for (int i=1; i<N; i++)
        if (arr[i] > arr[maxInd]) maxInd = i;
    return maxInd;
}
void minCashFlowRec(int amount[]) {
    int mxCredit = getMax(amount), mxDebit = getMin(amount);
    if (amount[mxCredit] == 0 && amount[mxDebit] == 0) return;
    int min = min(-amount[mxDebit], amount[mxCredit]);
    amount[mxCredit] -= min;
    amount[mxDebit] += min;
    cout << "Person " << mxDebit << " pays " << min
         << " to " << "Person " << mxCredit << endl;
    minCashFlowRec(amount);
}
void minCashFlow(int graph[][N]) {
    int amount[N] = {0};
    // calculated by subtracting debts of 'p' from credits of 'p'
    for (int p=0; p<N; p++)
       for (int i=0; i<N; i++)
          amount[p] += (graph[i][p] -  graph[p][i]);
    minCashFlowRec(amount);
}

//12>> Minimum Cost to cut a board into squares // TC o(nlogn)
int minimumCostOfBreaking(int X[], int Y[], int m, int n) {
    int res = 0;
    sort(X, X + m, greater<int>());
    sort(Y, Y + n, greater<int>());
    int hzntl = 1, vert = 1;
    int i = 0, j = 0;
    while (i < m && j < n) {
        if (X[i] > Y[j]){
            res += X[i] * vert;
            hzntl++;
            i++;
        }
        else{
            res += Y[j] * hzntl;
            vert++;
            j++;
        }
    }
    int total = 0;
    while (i < m)
        total += X[i++];
    res += total * vert;
    total = 0;
    while (j < n)
        total += Y[j++];
    res += total * hzntl;
    return res;
}

//13>>Find the minimum number of days on which you need to buy food
// from the shop so that you can survive the next S days // Tc o(1)
void survival(int S, int N, int M) {
    // If we can not buy at least a week
    // supply of food during the first week
    // OR We can not buy a day supply of food
    // on the first day then we can't survive.
    if (((N * 6) < (M * 7) && S > 6) || M > N)
        cout << "No\n";
    else {
        // If we can survive then we can
        // buy ceil(A/N) times where A is
        // total units of food required.
        int days = (M * S) / N;
        if (((M * S) % N) != 0)
            days++;
        cout << "Yes " << days << endl;
    }
}

//14>> Maximum product subset of an array // TC o(n)
int maxProductSubset(int a[], int n){
    if (n == 1) return a[0];
    int max_neg = INT_MIN;
    int count_neg = 0, count_zero = 0;
    int prod = 1;
    for (int i = 0; i < n; i++) {
        if (a[i] == 0){
            count_zero++;
            continue;
        }
        if (a[i] < 0){
            count_neg++;
            max_neg = max(max_neg, a[i]);
        }
        prod = prod * a[i];
    }
    if (count_zero == n) return 0;
    // If there are odd number of negative numbers
    if (count_neg & 1) {
        // Exceptional case: There is only
        // negative and all other are zeros
        if (count_neg == 1 && count_zero > 0 &&
            count_zero + count_neg == n) return 0;
        // Otherwise result is product of all non-zeros divided by
        //negative number with least absolute value
        prod = prod / max_neg;
    }
    return prod;
}

//15>> Maximize sum after K negations // TC o(nlogn)
long long int maximizeSum(long long int arr[], int n, int k){
    long long int sum=0;
    int i=0;
    sort(arr, arr+n);
    while (k>0){
        if (arr[i]==0) k=0;
        if(arr[i]>0) break;
        else{
            arr[i] = (-1)*arr[i];
            k--;
        }
        i++;
    }

    for(int j=0; j<n; j++){
        sum += arr[j];
    }
    if(k){
    sort(arr, arr+n);
    if(k&1) sum = sum - 2*arr[0];
    }
    return sum;
}

//16>> Maximize sum(arr[i]*i) of an Array // TC o(nlogn)
int Maximize(int a[],int n){
    sort(a,a+n);
    int mod=1e9+7;
    int sum=0;
    for(long i=0;i<n;i++)
    sum=(sum+a[i]*i)%mod;
    return sum;
}

//17>> Maximum sum of absolute difference of any permutation // TC o(nlogn)
int MaxSumDifference(int a[], int n) {
    vector<int> finalSequence;
    sort(a, a + n);
    for (int i = 0; i < n / 2; ++i) {
        finalSequence.push_back(a[i]);
        finalSequence.push_back(a[n - i - 1]);
    }
    if (n % 2 != 0) finalSequence.push_back(a[n/2]);
    int MaximumSum = 0;
    for (int i = 0; i < n - 1; ++i)
        MaximumSum = MaximumSum+abs(finalSequence[i]-finalSequence[i+1]);
    MaximumSum = MaximumSum+abs(finalSequence[n-1]-finalSequence[0]);
    return MaximumSum;
}

//18>> Swap and Maximize // same as above // TC o(nlogn)
sort(a,a+n);
	    int l=0,r=n-1,k=0;
	    long int sum=0;
	    while(l<r) {
	        sum+=(abs(a[l]-a[r]));
	        if(k%2==0) {
	            l++;
	            k=1;
	        }
	        else {
	            r--;
	            k=0;
	        }
	    }
	    cout<<sum+(abs(a[0]-a[r]))<<endl;

//19>> Minimum sum of absolute difference of pairs of two arrays
  long long int findMinSum(int a[], int b[], int n){
    sort(a, a+n);
    sort(b, b+n);
    long long int sum= 0 ;
    for (int i=0; i<n; i++)
        sum = sum + abs(a[i]-b[i]);
    return sum;
}

//20>> Page Faults in LRU // TC o(nn)
	    int n,k;
	    cin>>n;
	    int a[n];
	    for(int i=0;i<n;i++) cin>>a[i];
	    cin>>k;
	    int j=0,count=0;
	    vector<int> v;
	    vector<int>::iterator it;
	    for(int i=0;i<n;i++){
	        if(find(v.begin(),v.end(),a[i])!=v.end()){
	            it=find(v.begin(),v.end(),a[i]);
	            v.erase(it);
	            v.push_back(a[i]);
	        }
	        else if(j<k) {
	            v.push_back(a[i]);
	            count++;
	            j++;
	        }
	        else{
	            v.erase(v.begin());
	            v.push_back(a[i]);
	            count++;
	        }
	    }
	    cout<<count<<endl;

//21>> Smallest subset with sum greater than all other elements
// TC o(nlogn)
int minElements(int arr[], int n) {
    int halfSum = 0;
    for (int i = 0; i < n; i++)
        halfSum = halfSum + arr[i];
    halfSum = halfSum / 2;
    sort(arr, arr + n, greater<int>());
    int res = 0, curr_sum = 0;
    for (int i = 0; i < n; i++) {
        curr_sum += arr[i];
        res++;
        if (curr_sum > halfSum) return res;
    }
    return res;
}

//22>> smallest number with given sum of digits as S and
// number of digits as D // TC o(d)
string smallestNumber(int sumOfDigits, int digits){
        if (sumOfDigits > 9 * digits) return "-1";
        if (sumOfDigits == 0) return (digits == 1) ? "0" : "-1";
        int result = 0;
        sumOfDigits = sumOfDigits - 1;
        int rem;
        for (int i = 0; i < digits; i++){
          if (sumOfDigits > 9){
             rem = 9;
             sumOfDigits -= 9;
            }
          else{
             rem = sumOfDigits;
             sumOfDigits = 0;
            }

          result = result+(rem*(int)pow(10, i));
        }
      result += (int)pow(10,digits-1);
      string resultant = to_string(result);
      return resultant;
    }

//23>> Find maximum sum possible equal sum of three stacks
// TC o(n1+n2+n3)
int maxSum(int stack1[], int stack2[], int stack3[], int n1,
           int n2, int n3) {
    int sum1 = 0, sum2 = 0, sum3 = 0;
    for (int i = 0; i < n1; i++)
        sum1 += stack1[i];
    for (int i = 0; i < n2; i++)
        sum2 += stack2[i];
    for (int i = 0; i < n3; i++)
        sum3 += stack3[i];
    int top1 = 0, top2 = 0, top3 = 0;
    while (1) {
        if (top1 == n1 || top2 == n2 || top3 == n3) return 0;
        if (sum1 == sum2 && sum2 == sum3) return sum1;
        if (sum1 >= sum2 && sum1 >= sum3)
            sum1 -= stack1[top1++];
        else if (sum2 >= sum1 && sum2 >= sum3)
            sum2 -= stack2[top2++];
        else if (sum3 >= sum2 && sum3 >= sum1)
            sum3 -= stack3[top3++];
    }
}

//24>> Rearrange characters // TC o(n) // SC o(n)
        string s;
        cin>>s;
        int n = s.length();
        unordered_map<int,int> m;
        int max =0;
        for(int i=0; i<n; i++){
            m[s[i]]++;
            if(m[s[i]]>max) max=m[s[i]];
        }
        if(max<=n-max+1) cout<<"1"<<endl;
        else cout<<"0"<<endl;
    }

//25>> The cost to connect two ropes is equal to sum of their lengths
// minimize cost // TC o(nlogn) // SC o(n)
long long minCost(long long arr[], long long n) {
    priority_queue<long long, vector<long long>, greater<long long>> pq;
    for(long long i=0;i<n;i++)
        pq.push(arr[i]);
    long long cost = 0;
    while(pq.size() != 1){
        long long val = pq.top();
        pq.pop();
        val += pq.top();
        pq.pop();
        pq.push(val);
        cost += val;
    }
    return cost;
}

//26>>find the maximum difference of the number of 0s and the number of 1s
// (number of 0s – number of 1s) in the substrings of a string. // TC o(n)
int maxSubstring(string s){
	    int max_until = 0;
	    int ma = INT_MIN;
	    for(int i=0; i<s.length();i++){
	        int x = (s[i]=='0')?1:-1;
	        max_until += x;
	        if(max_until>ma) ma = max_until;
	        if(max_until<0) max_until = 0;
	    }
	    return ma;
	}

//27>> 
