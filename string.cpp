he
// string

//1>> reverse a string // TC o(n)
void reverseStr(string& str) {
    int n = str.length();
    for (int i = 0; i < n / 2; i++)
        swap(str[i], str[n - i - 1]);
}

//2>> check palindrome // TC o(n)
int isPlaindrome(string S){
    int l =S.size();
   for(int i=0; i<(S.size()/2); i++){
       l--;
       if(S[i]!=S[l]) return 0;
   }
   return 1;
}

//3>> Find Duplicate characters in a string // TC o(n)
void printDups(char *str) {
       map<char, int> count;
       for(int i=0; i<str.length(); i++)
       count[str[i]]++;
        int i;
        for (i = 0; i < NO_OF_CHARS; i++)
        if(count[i] > 1)
            printf("%c, count = %d \n", i, count[i]);
    }

//4>> Write a Code to check whether one string is a rotation of another
// TC o(n)
bool areRotations(string str1, string str2){
   if (str1.length() != str2.length())
        return false;
   string temp = str1 + str1;
  return (temp.find(str2) != string::npos);
}

//5>> Write a Program to check whether a string is a valid
// shuffle of two strings or not // TC o(n)
static boolean shuffleCheck(String first, String second, String result) {
   if(first.length() + second.length() != result.length())
     return false;
   int i = 0, j = 0, k = 0;
   while (k != result.length()) {
     if (i < first.length() && first[i] == result.[k])
       i++;
     else if (j < second.length() && second.[j] == result.[k])
       j++;
     else return false;
     k++;
   }
   if(i < first.length() || j < second.length()) return false;
   return true;
 }

 //6>> Count and say // TC o(nn)
 Input: n = 4
Output: "1211"
Explanation:
// countAndSay(1) = "1"
// countAndSay(2) = say "1" = one 1 = "11"
// countAndSay(3) = say "11" = two 1 = "21"
// countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"
 string countAndSay(int n) {
        if(n==1) return "1";
        if(n==2) return "11";
        string s = "11";
        for(int i=2; i<n; i++){
            string t = "";
            s+="&";
            int c=1;
            for(int j=1; j<s.length(); j++){
                if(s[j]!=s[j-1]){
                    t+=to_string(c);
                    t+=s[j-1];
                    c=1;
                }
                else c++;
            }
            s=t;
        }
        return s;
    }

//7>> Write a program to find the longest Palindrome in a string.
// [ Longest palindromic Substring] // TC o(nn)
void lcs(string str){
    int l=str.size();
    int i, j, max=1, start=0, k;
    bool table[l][l];
    for(i=0; i<l; i++)
    for(j=0; j<l; j++)
    table[i][j]=0;
    for(i=0; i<l; i++)
        table[i][i]=true;
    int r=0;
    for(i=0; i<l; i++)
        if(str[i]==str[i+1]){
            table[i][i+1]=true;
            if(r==0){
                start=i;
                max=2;
                r=1;
            }
        }
    for(k=3; k<=l; k++)
        for(i=0; i<l-k+1; i++){
            j=i+k-1;
            if(table[i+1][j-1] && str[i]==str[j]){
                table[i][j]=true;
                if(k>max){
                    start=i;
                    max=k;
                }
            }
        }
    print(str, start, start+max-1);
}

//8>> Longest Repeating Subsequence // TC o(nn) // SC o(nn)
int dp[n+1][n+1];
for(int i=0; i<=n; i++){
    for(int j=0; j<=n; j++){
        if(i==0 || j==0)
        dp[i][j]=0;
        else if(s[i-1]==s[j-1] && i!=j)
        dp[i][j]=dp[i-1][j-1]+1;
        else
        dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
    }
}
cout<<dp[n][n]<<endl;
}

//9>> Print all subsequences of a string // TC o(2^n)
void printSubSeqRec(string str, int n,
                    int index = -1, string curr = ""){
    if (index == n) return;
    if (!curr.empty()) cout << curr << "\n";
    for (int i = index + 1; i < n; i++) {
        curr += str[i];
        printSubSeqRec(str, n, i, curr);
        curr = curr.erase(curr.size() - 1);
    }
    return;
}
void printSubSeq(string str){
    printSubSeqRec(str, str.size());
}

//10>> Permutations of a given string
void permutations(char *ch,int l,int r){
    if(l==r) cout<<ch<<" ";
    else{
        for(int i=l;i<=r;i++){
            swap(ch+i,ch+l);
            sort(ch+l+1,ch+r+1);
            permutations(ch,l+1,r);
             sort(ch+l+1,ch+r+1);
            swap(ch+i,ch+l);
        }
    }
}
      int len=strlen(ch)-1;
	    sort(ch,ch+len+1);
	    permutations(ch,0,len);

//11>> Split the binary string into substrings with equal
// number of 0s and 1s // TC o(n)
int maxSubStr(string str, int n){
    int count0 = 0, count1 = 0;
    int cnt = 0;
    for (int i = 0; i < n; i++){
        if (str[i] == '0') count0++;
        else count1++;
        if (count0 == count1) cnt++;
    }
    if (count0 != count1) return -1;
    return cnt;
}

//12>> count the number of words in a single line with space
// character between two words // TC o(nn)
void print(int p[], int n){
    if(p[n]==1)
    cout<<p[n]<<" "<<n<<" ";
    else{
        print(p,p[n]-1);
        cout<<p[n]<<" "<<n<<" ";
    }
}
void word(int a[], int n, int k){
    int space[n+1][n+1];
    int ls[n+1][n+1];
    int c[n+1];
    int p[n+1];
    for(int i=1; i<=n; i++){
        space[i][i]=k-a[i];
        for(int j=i+1; j<=n; j++)
            space[i][j]=space[i][j-1]-a[j]-1;
    }
    for(int i=1; i<=n; i++){
        for(int j=i; j<=n; j++){
            if(space[i][j]<0) ls[i][j]=INT_MAX;
            else if(j==n && space[i][j]>=0) ls[i][j]=0;
            else ls[i][j]=space[i][j]*space[i][j];
        }
    }
    c[0]=0;
    for(int i=1; i<=n; i++){
        c[i]=INT_MAX;
        for(int j=1; j<=i; j++){
            if(c[j-1] != INT_MAX && ls[j][i] != INT_MAX && c[j-1]+ls[j][i]<c[i]){
                c[i]=c[j-1]+ls[j][i];
                p[i]=j;
            }
        }
    }
    print(p,n);
    cout<<endl;
}

//13>> Find the minimum number of operations that need to be performed
//on str1 to convert it to str2. The possible operations are:
// Insert
// Remove
// Replace // TC o(nn) // SC o(nn)
int dp[1001][1001];
	int fun(string s, string t, int m, int n){
	    if(m==-1) return n+1;
	    if(n==-1) return m+1;
	    if(dp[m][n]!=-1) return dp[m][n];
	    if(s[m]==t[n]) return dp[m][n] = fun(s,t,m-1,n-1);
	    int a = fun(s,t,m-1,n-1);
	    int b = fun(s,t,m,n-1);
	    int c = fun(s,t,m-1,n);
	    return dp[m][n]=1+min(a,min(b,c));
	}
		int editDistance(string s, string t){
		    memset(dp,-1,sizeof(dp));
		    int m = s.length();
		    int n = t.length();
		    return fun(s,t,m-1,n-1);
		}

//14>> Lexicographically next greater permutation of list of numbers
// TC o(n) // SC o(n)
int  idx =-1;
for(int i=n-1; i>0; i--){
    if(arr[i]>arr[i-1]){
        idx=i;
        break;
    }
}
if(idx==-1) reverse(arr,arr+n);
else{
    int prev =idx;
    for(int i=idx+1; i<n; i++){
        if(arr[i]>arr[idx-1] && arr[i]<=arr[prev]){
            prev = i;
        }
    }
    swap(arr[idx-1],arr[prev]);
    reverse(arr+idx,arr+n);
}
for(int i=0; i<n; i++)
  cout<<arr[i]<<" ";

//15>> Parenthesis Checker // TC o(x) // SC o(x)
bool ispar(string x){
    int i =0;
    int j =0;
    int k =0;
    for(int v=0; v<x.length(); v++){
        char c =x[v];
        if(c=='(')
        i++;
        else if(c=='{')
        j++;
        else if(c=='[')
        k++;
        else if(c==')'){
            if(i<=0)
            return false;
            i--;
        }
        else if(c=='}'){
            if(j<=0)
            return false;
            j--;
        }
        else if(c==']'){
            if(k<=0)
            return false;
            k--;
        }
    }
    return (i==0 && j==0 && k==0);
}

//16>> find out if A can be segmented into a space-separated sequence
// of dictionary words // TC o(ss) // SC o(s)
unordered_map<string,int> dp;
int solve(string s,vector<string> &b){
    int sz = s.length();
    if(sz==0) return 1;
    if(dp[s]!=0) return dp[s];
    for(int i=1; i<=sz; i++){
        int f=0;
        string ss = s.substr(0,i);
        for(int j=0; j<b.size(); j++){
            if(ss.compare(b[j])==0){
                f=1; break;
            }
        }
        if(f==1 && solve(s.substr(i,sz-i),b)==1)
        return dp[s]=1;
    }
    return dp[s]=-1;
}

int wordBreak(string A, vector<string> &B) {
    int x = solve(A,B);
    if(x==1) return 1;
    else return 0;
}

//17>> Rabin-Karp Algorithm for Pattern Searching
// TC omega(m+n) , o(mn)
void search(char pat[], char txt[], int q)  {
    int M = strlen(pat);
    int N = strlen(txt);
    int i, j;
    int p = 0;
    int t = 0;
    int h = 1;
    for (i = 0; i < M - 1; i++)
        h = (h * d) % q;
    for (i = 0; i < M; i++)  {
        p = (d * p + pat[i]) % q;
        t = (d * t + txt[i]) % q;
    }
    for (i = 0; i <= N - M; i++)  {
        if ( p == t ){
            for (j = 0; j < M; j++)
                if (txt[i+j] != pat[j]) break;
            if (j == M)
                cout<<"Pattern found at index "<< i<<endl;
        }
        if ( i < N-M ){
            t = (d*(t - txt[i]*h) + txt[i+M])%q;
            if (t < 0)  t = (t + q);
        }
    }
}

//18>> KMP Algo // Longest Prefix Suffix // TC o(s) // SC o(s)
int lps(string s){
	    int n =s.length();
	    int arr[n];
	    int i=1; int j=0;
	    arr[0]=0;
	    while(i<n){
	        if(s[i]==s[j]){
	            arr[i]=j+1;
	            i++;
	            j++;
	        }
	        else{
	            if(j != 0) j = arr[j-1];
	            else{
	                arr[i] = 0;
	                i++;
	            }
	        }
	    }
	    return arr[n-1];
}

//19>> Convert a Sentence into its equivalent mobile numeric keypad sequence.
// TC o(n)
string printSequence(string arr[],string input) {
    string output = "";
    int n = input.length();
    for (int i=0; i<n; i++) {
        if (input[i] == ' ') output = output + "0";
        else{
            int position = input[i]-'A';
            output = output + arr[position];
        }
    }
    return output;
}

//20>> brackets '{' and '}' find out the minimum number of reversals
// required to make a balanced expression // TC o(n)
int n=s.length();
	    if(n&1) cout<<"-1"<<endl;
	    else{
	            int l=0,c=0;
	        for(int i=0;i<n;i++){
	            if(s[i]=='{')
	            l++;
	            if(s[i]=='}')
	            l--;

	            if(l<0){
	                l=1;
	                c++;
	            }

	        }
	        c=c+l/2;
	        cout<<c<<endl;
	    }

//21>> Count Palindromic Subsequences // TC o(nn) // SC o(nn)
int dp[1001][1001];
int fun(int i, int j, string s){
    if(i>j) return 0;
    if(i==j) return 1;
    if(dp[i][j]!=-1) return dp[i][j];
    if(s[i]==s[j]){
        return dp[i][j] = fun(i+1,j,s) + fun(i,j-1,s) + 1;
    }
    else{
        return dp[i][j] = fun(i+1,j,s) +fun(i,j-1,s) - fun(i+1,j-1,s);
    }
}
int countPS(string str){
   int n = str.length();
   dp[n][n];
   memset(dp,-1,sizeof(dp));
   return fun(0,n-1,str);
}

//22>> Count of number of given string in 2D character array
// TC o(nn)
int internalSearch(string needle, int row,
                   int col, string hay[],
                   int row_max, int col_max, int xx)  {
    int found = 0;
    if (row >= 0 && row <= row_max && col >= 0 &&
        col <= col_max && needle[xx] == hay[row][col]){
        char match = needle[xx];
        xx += 1;
        hay[row][col] = 0;
        if (needle[xx] == 0) found = 1;
        else{
            found += internalSearch(needle, row,
                                    col + 1, hay,
                                    row_max, col_max,xx);
            found += internalSearch(needle, row, col - 1,
                                    hay, row_max, col_max,xx);
            found += internalSearch(needle, row + 1, col,
                                    hay, row_max, col_max,xx);
            found += internalSearch(needle, row - 1, col,
                                    hay, row_max, col_max,xx);
        }
        hay[row][col] = match;
    }
    return found;
}
int searchString(string needle, int row, int col,
                  string str[], int row_count,
                                int col_count){
    int found = 0;
    int r, c;
    for (r = 0; r < row_count; ++r) {
        for (c = 0; c < col_count; ++c){
            found += internalSearch(needle, r, c, str,
                                    row_count - 1,
                                    col_count - 1, 0);
        }
    }
    return found;
}

//23>> Find the string in grid // TC o(mn)
int xdir[8]={0,-1,-1,-1,0,1,1,1};
int ydir[8]={1,1,0,-1,-1,-1,0,1};
bool find(vector<vector<char>> &mat, int x,int y, int n, int m, string str){
    for(int i=0;i<8;i++){
        int X=x+xdir[i];
        int Y=y+ydir[i];
        int j=1;
        if(j==str.length()) return true;
        while(X>=0 && Y>=0 && X<n && Y<m){
            if(mat[X][Y]==str[j]){
                   j++;
                   if(j==str.length()) return true;
                 X+=xdir[i];
                 Y+=ydir[i];
              }
            else break;
        }
    }
    return false;
}
void solution(vector<vector<char>> &mat, int n, int m, string str){
    int flag=0;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(str[0]==mat[i][j] && find(mat,i,j,n,m,str)){
                cout<<i<<" "<<j<<", ";
                flag=1;
            }
        }
    }
    if(flag==0) cout<<-1;
    cout<<endl;
}

//24>> Boyer Moore Algorithm for Pattern Searching // TC
void badCharHeuristic( string str, int size,
                        int badchar[NO_OF_CHARS]){
    int i;
    for (i = 0; i < NO_OF_CHARS; i++)
        badchar[i] = -1;
    for (i = 0; i < size; i++)
        badchar[(int) str[i]] = i;
}
void search( string txt, string pat)  {
    int m = pat.size();
    int n = txt.size();
    int badchar[NO_OF_CHARS];
    badCharHeuristic(pat, m, badchar);
    int s = 0;
    while(s <= (n - m)){
        int j = m - 1;
        while(j >= 0 && pat[j] == txt[s + j])
            j--;
        if (j < 0)  {
            cout << "pattern occurs at shift = " <<  s << endl;
            s += (s + m < n)? m-badchar[txt[s + m]] : 1;
        }
        else
            s += max(1, j - badchar[txt[s + j]]);
    }
}

//25>> Roman Number to Integer // TC o(s)
int romanToDecimal(string &str){
   unordered_map<char, int> mp;
    mp['I'] = 1;
    mp['V'] = 5;
    mp['X'] = 10;
    mp['L'] = 50;
    mp['C'] = 100;
    mp['D'] = 500;
    mp['M'] = 1000;
    int res = 0;
    for(int i = 0; i < str.size(); i++){
        if(i+1 < str.size() && mp[str[i]] < mp[str[i+1]]){
            res += mp[str[i+1]] - mp[str[i]];
            i += 1;
        }
        else res += mp[str[i]];
    }
    return res;
}

//26>> Longest Common Prefix // TC o(nn)
string longestCommonPrefix(vector<string>& v) {
        int mi = INT_MAX;
        if(v.size()==0) return "";
        string c = v[0];
        for(int i =0; i<v.size(); i++){
            int j=0; int k=0; int a=0;
            while(j<c.size() && k<v[i].size()){
                if(c[j]==v[i][k]) a++;
                else break;
                j++; k++;
            }
            mi = min(mi,a);
        }
         return c.substr(0,mi);
    }

//27>> make this string a sequence of alternate characters by flipping
// some of the bits, our goal is to minimize the number of bits
// to be flipped // TC o(n)
     int c1=0;
     int c2=0;
     string s;
     cin>>s;
     for(int i=0; i<s.length(); i++){
         if(i&1 and s[i]=='0') c1++;
         if(i%2==0 and s[i]=='1') c1++;
         if(i&1 and s[i]=='1') c2++;
         if(i%2==0 and s[i]=='0') c2++;
     }
     cout<<min(c1,c2)<<endl;

//28>> Second most repeated string in a sequence
// TC o(n*max(s)) // SC o(n*max(s))
string secFrequent (string arr[], int n){
    unordered_map<string, int> occ;
    for (int i = 0; i < n; i++)
        occ[arr[i]]++;
    int first_max = INT_MIN, sec_max = INT_MIN;
    for (auto it = occ.begin(); it != occ.end(); it++) {
        if (it->second > first_max) {
            sec_max = first_max;
            first_max = it->second;
        }
        else if (it->second > sec_max &&
                 it->second != first_max)
            sec_max = it->second;
    }
    for (auto it = occ.begin(); it != occ.end(); it++)
        if (it->second == sec_max)
            return it->first;
    }

//29>> Minimum Swaps for Bracket Balancing // TC o(nn)
      int n;
	    cin>>n;
	    string s;
	    cin>>s;
	    stack<char> stk;
	    int count=0;
	    for(int i=0;i<n;i++){
	        if(s[i]=='[') stk.push(s[i]);
	        else{
	            if(stk.empty()){
	                int j=i+1;
	                while(s[j]!='[') j++;
	                while(i!=j){
	                    count++;
	                    swap(s[j],s[j-1]);
	                    j--;
	                }
	                stk.push(s[i]);
	            }
	            else stk.pop();
	        }
	    }
	    cout<<count<<endl;

//30>> Longest Common Subsequence // o(s1s2)
int lcs(int x, int y, string s1, string s2){
    int dp[x+1][y+1];
    for(int i = 0; i <= x; i++)
        dp[i][0] = 0;
    for(int j = 0; j <= y; j++)
        dp[0][j] = 0;
    for(int i = 1; i <= x; i++){
        for(int j = 1; j <= y; j++){
            if(s1[i-1] == s2[j-1])
                dp[i][j] = 1 + dp[i-1][j-1];
            else
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[x][y];
}

//31>> Program to generate all possible valid IP addresses from
// given string // TC o(nnn) // SC o(n)
int is_valid(string ip){
    vector<string> ips;
    string ex = "";
    for (int i = 0; i < ip.size(); i++) {
        if (ip[i] == '.') {
            ips.push_back(ex);
            ex = "";
        }
        else ex = ex + ip[i];
    }
    ips.push_back(ex);
    for (int i = 0; i < ips.size(); i++) {
        if (ips[i].length() > 3
            || stoi(ips[i]) < 0 || stoi(ips[i]) > 255)
            return 0;
        if (ips[i].length() > 1 && stoi(ips[i]) == 0)
            return 0;
        if (ips[i].length() > 1
            && stoi(ips[i]) != 0 && ips[i][0] == '0')
            return 0;
    }
    return 1;
}
void convert(string ip){
    int l = ip.length();
    if (l > 12 || l < 4)
        cout << "Not Valid IP Address";
    string check = ip;
    vector<string> ans;
    for (int i = 1; i < l - 2; i++) {
        for (int j = i + 1; j < l - 1; j++) {
            for (int k = j + 1; k < l; k++) {
                check = check.substr(0, k) + "."
                        + check.substr(k, l - k + 2);
                check = check.substr(0, j) + "."
                      + check.substr(j, l - j + 3);
                check = check.substr(0, i) + "."
                      + check.substr(i, l - i + 4);
                if (is_valid(check)) {
                    ans.push_back(check);
                    std::cout << check << '\n';
                }
                check = ip;
            }
        }
    }
}

//32>> find the smallest window length that contains all the characters
// of the given string at least one time // TC o(256n) // SC o(256)
string findSubString(string str){
   int n = str.length();
    int dist_count = 0;
    bool visited[no_of_chars] = { false };
    for (int i = 0; i < n; i++) {
        if (visited[str[i]] == false) {
            visited[str[i]] = true;
            dist_count++;
        }
    }
    int start = 0, start_index = -1, min_len = INT_MAX;
    int count = 0;
    int curr_count[no_of_chars] = { 0 };
    for (int j = 0; j < n; j++) {
        curr_count[str[j]]++;
        if (curr_count[str[j]] == 1) count++;
        if (count == dist_count) {
            while (curr_count[str[start]] > 1) {
                if (curr_count[str[start]] > 1)
                    curr_count[str[start]]--;
                start++;
            }
            int len_window = j - start + 1;
            if (min_len > len_window) {
                min_len = len_window;
                start_index = start;
            }
        }
    }
    return str.substr(start_index, min_len);
    }

//33>> rearrange characters in a string such that no two adjacent
// characters are same // TC o(n) // SC o(n)
int n = s.length();
        unordered_map<int,int> m;
        int max =0;
        for(int i=0; i<n; i++){
            m[s[i]]++;
            if(m[s[i]]>max) max=m[s[i]];
        }
        if(max<=n-max+1) cout<<"1"<<endl;
        else cout<<"0"<<endl;

//34>> Minimum characters to be added at front to make string palindrome
// TC o(n) using KMP
vector<int> computeLPSArray(string str) {
    int M = str.length();
    vector<int> lps(M);
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M) {
        if (str[i] == str[len]){
            len++;
            lps[i] = len;
            i++;
        }
        else{
            if (len != 0) len = lps[len-1];
            else{
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}
int getMinCharToAddedToMakeStringPalin(string str){
    string revStr = str;
    reverse(revStr.begin(), revStr.end());
    string concat = str + "$" + revStr;
    vector<int> lps = computeLPSArray(concat);
    return (str.length() - lps.back());
}

//35>> Print Anagrams Together // TC o(nslogs) // SC o(ns)
vector<vector<string> > Anagrams(vector<string>& a) {
    map<string,vector<string>> m;
    for(int i=0; i<a.size(); i++){
        string s= a[i];
        sort(s.begin(),s.end());
        m[s].push_back(a[i]);
    }
    vector<vector<string>> ans(m.size());
    int idx=0;
    for(auto x:m){
        auto v=x.second;
        for(int i=0; i<v.size(); i++){
            ans[idx].push_back(v[i]);
        }
        idx++;
    }
    return ans;

//36>> Smallest window in a string containing all the characters
// of another string // TC o(s)
string smallestWindow (string s, string pat){
  int str[256]={0},patt[256]={0};
    for(int i=0;i<pat.length();i++)
        patt[pat[i]]++;
    int i=0,j=0;
    int end=s.length();
    string res="";
    int c=0;
    while(j!=end){
        while(c!=pat.length() && j!=end){
            if(patt[s[j]]!=0){
                str[s[j]]++;
                if(str[s[j]]<=patt[s[j]]) c++;
            }
            j++;
        }
        string temp="";
        while(c==pat.length()){
            temp=s.substr(i,j-i);
            if(temp.length()<res.length() || res=="") res= temp;
            if(patt[s[i]]!=0){
                if(str[s[i]]==patt[s[i]]) c--;
                str[s[i]]--;
            }
            i++;
        }
    }
    return res==""?"-1":res;
}

//37>> Remove Consecutive Characters // TC o(s) // SC o(s)
string removeConsecutiveCharacter(string s){
        string ans ="";
        char temp;
        for(int i=0; i<s.length(); i++){
            if(temp!=s[i]) ans+=s[i];
            temp=s[i];
        }
        return ans;
    }

//38>> Wildcard string matching
// * --> Matches with 0 or more instances of any character
//       or set of characters.
// ? --> Matches with any one character. // TC o(nn)
int dp[1001][1001];
int solve(string p, string s, int i, int j){
    if(i==-1 && j==-1) return 1;
    if(j==-1){
        for(int k=0; k<i; k++){
            if(p[k]!='*') return 0;
        }
        return 1;
    }
    if(i==-1) return 0;
    if(dp[i][j]!=-1) return dp[i][j];
    if(p[i]==s[j] || p[i]=='?') return dp[i][j] = solve(p,s,i-1,j-1);
    if(p[i]=='*'){
        int opt1 = solve(p,s,i-1,j);
        int opt2 = solve(p,s,i,j-1);
        return dp[i][j] = opt1 || opt2;
    }
    return dp[i][j] = 0;
}

//39>> Function to find Number of customers who could not get a computer
// TC o(n) // SC(MAX_CHAR)
int runCustomerSimulation(int n, const char *seq) {
    char seen[MAX_CHAR] = {0};
    int res = 0;
    int occupied = 0;
    for (int i=0; seq[i]; i++){
        int ind = seq[i] - 'A';
        if (seen[ind] == 0) {
            seen[ind] = 1;
            if (occupied < n) {
                occupied++;
                seen[ind] = 2;
            }
            else res++;
        }
        else{
           if (seen[ind] == 2) occupied--;
           seen[ind] = 0;
        }
    }
    return res;
}

//40>> Transform One String to Another using Minimum Number
// of Given Operation // TC o(n)
int minOps(string& A, string& B) {
    int m = A.length(), n = B.length();
    if (n != m) return -1;
    int count[256];
    memset(count, 0, sizeof(count));
    for (int i=0; i<n; i++)
       count[B[i]]++;
    for (int i=0; i<n; i++)
       count[A[i]]--;
    for (int i=0; i<256; i++)
      if (count[i]) return -1;
    int res = 0;
    for (int i=n-1, j=n-1; i>=0; ){
        while (i>=0 && A[i] != B[j]){
            i--;
            res++;
        }
        if (i >= 0){
            i--;
            j--;
        }
    }
    return res;
}

//41>> Isomorphic Strings [if there is a one to one mapping possible
// for every character of str1 to every character of str2 while
// preserving the order] // TC o(s1+s2) // SC o(MAX_CHAR)
bool areIsomorphic(string s1, string s2){
    int n= s1.length();
    int ma = s2.length();
    if(n!=ma) return false;
    int m1[256] = {0};
    int m2[256] = {0};
    for(int i=0; i<n; i++){
        if(!m1[s1[i]] && !m2[s2[i]]){
            m1[s1[i]] = s2[i];
            m2[s2[i]] = s1[i];
        }
        else if(m1[s1[i]] != s2[i]) return false;
    }
    return true;
}

//42>> Recursively print all sentences that can be formed from
// list of word lists // TC o(nn)
void printUtil(string arr[R][C], int m, int n, string output[R]){
    output[m] = arr[m][n];
    if (m==R-1){
        for (int i=0; i<R; i++)
           cout << output[i] << " ";
        cout << endl;
        return;
    }
    for (int i=0; i<C; i++)
       if (arr[m+1][i] != "")
          printUtil(arr, m+1, i, output);
}
void print(string arr[R][C]){
   string output[R];
   for (int i=0; i<C; i++)
     if (arr[0][i] != "")
        printUtil(arr, 0, i, output);
}
