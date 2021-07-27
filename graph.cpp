
//Graph

//1>> bfs // TC o(v+e) // SC o(v)
vector<int>bfsOfGraph(int V, vector<int> g[]){
	    vector<int> v;
	    queue<int> q;
	    q.push(0);
	    vector<int> vis(V,0);
	    vis[0] = 1;
	    while(!q.empty()){
	        int td  = q.front();
	        q.pop();
	        v.push_back(td);
	        for(auto x:g[td]){
	            if(!vis[x]){
	                vis[x] = 1;
	                q.push(x);
	            }
	        }
	    }
	    return v;
	}

//2>> dfs // TC o(v+e) // SC o(v)
vector<int> v;
 void solve(int s, vector<int> &vis, vector<int> g[]){
     vis[s] = 1;
     v.push_back(s);
     for(auto x:g[s]){
         if(!vis[x]) solve(x,vis,g);
     }
 }
	vector<int>dfsOfGraph(int V, vector<int> adj[]){
	    v.clear();
	    vector<int> vis(V,0);
	    solve(0,vis,adj);
	    return v;
	}

//3>>  Detect cycle in a directed graph // TC o(e+v) // SC o(v)
bool solve(int src, vector<int> &vis, vector<int> &order, vector<int> adj[]){
        vis[src] = 1;
        order[src] = 1;
        for(auto x:adj[src]){
            if(!vis[x]){
                bool conf = solve(x,vis,order,adj);
                if(conf) return true;
            }
            else if(order[x]) return true;
        }
        order[src] = 0;
        return false;
    }
	bool isCyclic(int v, vector<int> adj[]) {
	   	vector<int> vis(v,0);
	   	vector<int> order(v,0);
	   	for(int i=0; i<v; i++){
	   	    bool c = solve(i,vis,order,adj);
	   	    if(c) return true;
	   	}
	   	return false;
	}

//4>> Detect cycle in an undirected graph // TC o(v+e) // SC o(v)
bool solve(int src, vector<bool> &vis, int par, vector<int> adj[]){
        vis[src] = true;
        for(auto x:adj[src]){
            if(!vis[x]){
                bool conf = solve(x,vis,src,adj);
                if(conf) return true;
            }
            else if(x!=par) return true;
        }
        return false;
    }
	bool isCycle(int v, vector<int>adj[]){
	    vector<bool> vis(v,false);
	    for(int i=0; i<v; i++){
	        if(!vis[i]){
	   	    bool c = solve(i,vis,-1,adj);
	   	    if(c) return true;
	        }
	   	}
	   	return false;
	}

//5>> Search in a Maze // TC o(nnnnnnnn) // SC o(nn)
void solve(int m[MAX][MAX],int i,int j,int n,vector<string>&res,string path){
    if(i<0 || i>=n || j<0 || j>=n || m[i][j]!=1)
    return;
    if(i==n-1 && j==n-1){
        res.push_back(path);
        return;
    }
    m[i][j]=-1;
    solve(m,i+1,j,n,res,path+'D');
    solve(m,i-1,j,n,res,path+'U');
    solve(m,i,j+1,n,res,path+'R');
    solve(m,i,j-1,n,res,path+'L');
    m[i][j]=1;
}
vector<string> findPath(int m[MAX][MAX], int n) {
    vector<string>res;
    string s="";
    solve(m,0,0,n,res,s);
    sort(res.begin(),res.end());
    return res;
}

//6>> Find out the minimum steps a Knight will take to reach the
// target position // TC o(nn) // SC o(nn)
int minStepToReachTarget(vector<int>&knightPos, vector<int>&targetPos, int N){
	    int res=0;
			bool visited[N*N];
			fill(visited,visited+N*N,false);
			pair<int,int> start,target;
			start=make_pair(knightPos[0],knightPos[1]);
			target=make_pair(targetPos[0],targetPos[1]);
			if(start.first==target.first && start.second==target.second)
					return res;
			queue<pair<int,int>> q;
			q.push(start);
			q.push({-1,-1});
			while(q.size()>1){
							pair<int,int> temp=q.front();
							q.pop();
							if(temp.first==-1 && temp.second==-1){
									res++;
									q.push({-1,-1});
							}else if(temp.first==target.first && temp.second==target.second){
							 		return res;
							}else{
									int x=temp.first;
									int y=temp.second;
									if(x+1<=N && y+2<=N && visited[(x+1-1)*N+y+2-1]==false){
									visited[(x+1-1)*N+y+2-1]=true;
									q.push({x+1,y+2});
									}
									if(x+1<=N && y-2> 0 && visited[(x+1-1)*N+y-2-1]==false){
									visited[(x+1-1)*N+y-2-1]=true;
									q.push({x+1,y-2});
									}
									if(x+2<=N && y+1<=N && visited[(x+2-1)*N+y+1-1]==false){
									visited[(x+2-1)*N+y+1-1]=true;
									q.push({temp.first+2,y+1});
									}
									if(x+2<=N && y-1> 0 && visited[(x+2-1)*N+y-1-1]==false){
									visited[(x+2-1)*N+y-1-1]=true;
									q.push({x+2,y-1});
									}
									if(x-1> 0 && y+2<=N && visited[(x-1-1)*N+y+2-1]==false){
									visited[(x-1-1)*N+y+2-1]=true;
									q.push({x-1,y+2});
									}
									if(x-1> 0 && y-2> 0 && visited[(x-1-1)*N+y-2-1]==false){
									visited[(x-1-1)*N+y-2-1]=true;
									q.push({x-1,y-2});
									}
									if(x-2> 0 && y-1> 0 && visited[(x-2-1)*N+y-1-1]==false){
									visited[(x-2-1)*N+y-1-1]=true;
									q.push({x-2,y-1});
									}
									if(x-2> 0 && y+1<=N && visited[(x-2-1)*N+y+1-1]==false){
									visited[(x-2-1)*N+y+1-1]=true;
									q.push({x-2,y+1});
									}
					}
			}
			return -1;
	}

//7>> flood fill", consider the starting pixel, plus any pixels connected
// 4-directionally to the starting pixel of the same color as the
// starting pixel // TC(e+v)
vector<vector<int>> ff(vector<vector<int>>& img, int sr, int sc, int nc) {
        int oldColor = img[sr][sc];
        if (oldColor != nc)
            dfs(img, sr, sc, oldColor, nc);
        return img;
    }
    void dfs(vector<vector<int>> &img, int x, int y, int &oldColor, int &nc){
        if (x<0 || x>=img.size() || y<0 ||
				 y>=img[0].size() || img[x][y] != oldColor) return;
        img[x][y] = nc;
        dfs(img, x + 1, y, oldColor, nc);
        dfs(img, x - 1, y, oldColor, nc);
        dfs(img, x, y + 1, oldColor, nc);
        dfs(img, x, y - 1, oldColor, nc);
    }

//8>> Clone a graph // TC(e+v) // SC(v)
void dfs(Node* node, Node* copy, vector<Node*> &vis){
        vis[copy->val] = copy;
        for(auto x:node->neighbors){
            if(vis[x->val]==NULL){
            Node* newnode = new Node(x->val);
            (copy->neighbors).push_back(newnode);
            dfs(x,newnode,vis);
            }
            else (copy->neighbors).push_back(vis[x->val]);
        }
    }
    Node* cloneGraph(Node* node){
        if(node == NULL) return NULL;
        vector<Node*> vis(1000,NULL);
        Node* copy = new Node(node->val);
        dfs(node,copy,vis);
        return copy;

    }

//9>> Topological sort // TC o(v+e) // SC o(v)
vector<int> topoSort(int V, vector<int> adj[]) {
	    vector<int> in(V,0);
	    queue<int> q;
	    vector<int> ans;
	    for(int i=0; i<V; i++)
	        for(auto x:adj[i]) in[x]++;
	    for(int i=0; i<V; i++)
	        if(in[i] == 0) q.push(i);
	    while(!q.empty()){
	        int td = q.front();
	        q.pop();
	        ans.push_back(td);
	        for(auto x:adj[td]){
	            in[x]--;
	            if(in[x]==0) q.push(x);
	        }
	    }
	    return ans;
	}

//10>> Negative weight cycle belmonford // TC o(ev)
    struct edge{
        int a, b, cost;
  };
		int isNegativeWeightCycle(int n, vector<vector<int>>edges){
		vector<edge>Edges;
		for(auto i: edges){
		edge p;
		p.a = i[0];
		p.b = i[1];
		p.cost = i[2];
		Edges.push_back(p);
	}
		vector<int> d(n);
		for (int i = 0; i < n; ++i) {
		for (edge e : Edges) {
		if (d[e.a] + e.cost < d[e.b])
		d[e.b] = d[e.a] + e.cost;
		}
 }
		for (edge e : Edges)
		if (d[e.a] + e.cost < d[e.b]) return 1;
		return 0;
}

//11>> Minimum time taken by each job to be completed given by a Directed
// Acyclic Graph
void addEdge(int u, int v){
    graph[u].push_back(v);
    indegree[v]++;
}
void printOrder(int n, int m){
    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (indegree[i] == 0) {
            q.push(i);
            job[i] = 1;
        }
    }
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        for (int adj : graph[cur]) {
            indegree[adj]--;

            // Push its adjacent elements
            if (indegree[adj] == 0)
                job[adj] = job[cur] + 1;
            q.push(adj);
        }
    }

    // Print the time to complete
    // the job
    for (int i = 1; i <= n; i++)
        cout << job[i] << " ";
    cout << "\n";
}

//12>> Find the number of islands // TC o(mn) // SC o(mn)
void dfs(int n, int m, int i, int j, vector<vector<char>> &grid){
       if(i<0 || j<0 || i>=n || j>=m || grid[i][j]=='0') return;
        grid[i][j]='0';
        dfs(n,m,i+1,j,grid);
        dfs(n,m,i,j+1,grid);
        dfs(n,m,i-1,j,grid);
        dfs(n,m,i,j-1,grid);
        dfs(n,m,i+1,j+1,grid);
        dfs(n,m,i-1,j-1,grid);
        dfs(n,m,i+1,j-1,grid);
        dfs(n,m,i-1,j+1,grid);
}
    int numIslands(vector<vector<char>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        int c=0;
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                if(grid[i][j]=='1'){
                    dfs(n,m,i,j,grid);
                    c++;
                }
            }
        }
        return c;
    }
//visited array is managed in grid array

//13>> Given a sorted Dictionary of an Alien Language, find order of characters
// TC o(n+k) // SC o(k)
void DFS(int x, vector<int> adj[], vector<bool> &visited, stack<int> & st){
    visited[x]=true;
    for(auto v: adj[x])
    if(!visited[v])
    DFS(v,adj,visited,st);
    st.push(x);
}
string findOrder(string dict[], int N, int K) {
   vector<int> adj[26];
   for(int i=1;i<N;i++){
       string str1= dict[i-1];
       string str2= dict[i];
       for(int j=0;j<min(str1.length(), str2.length());j++){
           if(str1[j]!=str2[j]){
               adj[str1[j]-'a'].push_back(str2[j]-'a');
               break;
           }
       }
   }
   stack<int> st;
   vector<bool> visited(26,false);
   for(int i=0;i<26;i++){
       if(adj[i].size()>0 && visited[i]==false)
       DFS(i, adj, visited, st);
   }
   string ans="";
   while(!st.empty()){
       char ch= 'a'+st.top();
       ans+=ch;
       st.pop();
   }
   return ans;
}

//14>> Strongly Connected Components (Kosaraju's Algo) // TC o(v+e) // SC o(v)
void DFSRec(vector<int> adj[], int s, stack<int> &st, bool visited[]){
	visited[s] = true;
	for(int u : adj[s]){
			if(visited[u] == false){
					DFSRec(adj, u, st, visited);
			}
	}
	st.push(s);
}

void DFS(vector<int> adj[], int V, stack<int> &st){
	bool visited[V];
	fill(visited, visited+V, false);
	for(int i = 0; i < V; i++){
			if(visited[i] == false){
					DFSRec(adj, i, st, visited);
			}
	}
}

void DFSRec1(vector<int> revadj[], int s, bool visited[]){
	visited[s] = true;
	for(int u : revadj[s]){
			if(visited[u] == false){
					DFSRec1(revadj, u, visited);
			}
	}
}

int kosaraju(int V, vector<int> adj[]){
	stack<int> st;
	DFS(adj, V, st);
	//Reverse Edges
	vector<int> revadj[V];
	for(int v = 0; v < V; v++){
			for(int u: adj[v]){
					revadj[u].push_back(v);
			}
	}
	//printStack(st);
	int count = 0;
	bool visited[V];
	fill(visited, visited+V, false);
	while(!st.empty()){
			int v = st.top();
			st.pop();
			if(visited[v] == false){
					DFSRec1(revadj, v, visited);
					count++;
			}
	}
	return count;
}

//15>> Check whether a given graph is Bipartite or not // TC o(v+e)
bool bfs(int u, vector<int> &color, vector<int> adj[]) {
  queue<int> q;
  q.push(u);
  color[u] = 1;
	while(!q.empty()) {
		int i = q.front(); q.pop();
		for(auto x : adj[i]) {
			if(color[x] == 0) {
				color[x] = -color[i];
				q.push(x);
			}
			else if(color[x] == color[i]) return false;
		}
	}
	return true;
	}
	bool isBipartite(int V, vector<int>adj[]){
		vector<int> color(V);
		for(int i = 0; i < V; i++) {
			if(!color[i] and !bfs(i, color, adj))
			return false;
		}
		return true;
}

//16>> Cheapest Flights Within K Stops // TC o(k(e+v)) // SC o(n)
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        unordered_map<int, vector<pair<int,int>> > mp;
        for(auto fl:flights) mp[fl.at(0)].push_back({fl.at(1), fl.at(2)});
        using VE=vector<int>;
        priority_queue< VE, vector<VE>, greater<VE>> pq;
        pq.push({0, src, 0});
        while(!pq.empty()){
            auto cur = pq.top(); pq.pop();
            int cost = cur.at(0), id=cur.at(1), nstop=cur.at(2);
            if(id==dst) return cost;
            if(nstop>K) continue;
            for(auto next: mp[id]){
                int idn = next.first, dcost = next.second;
                pq.push({cost+dcost, idn, nstop+1});
            }
        }
        return -1;
    }

//17>> Journey to the Moon // TC  o(e+v) // SC o(v)
void dfs(int src,vector<int> &vis,vector<int> g[],int &counter){
    vis[src] = 1;
    counter++;
    for(auto x:g[src]){
        if(!vis[x]) dfs(x,vis,g,counter);
    }
}
int32_t main(){
    int v,e;
    cin>>v>>e;
    vector<int> g[v];
    for(int i=0;i<e;i++){
        int x,y;
        cin>>x>>y;
        g[x].push_back(y);
        g[y].push_back(x);
    }
    vector<int> solution;
    vector<int> vis(v,0);
    for(int i=0;i<v;i++){
        if(!vis[i]){
            int counter = 0;
            dfs(i,vis,g,counter);
            solution.push_back(counter);
        }
    }
    int val = (v*(v-1)) / 2;
    for(int i=0;i<solution.size();i++){
        int x = (solution[i]*(solution[i]-1)) / 2;
        val = val - x;
    }
    cout<<val;
    return 0;
}

//18>> Oliver and the Game // TC o(e+v)
#include<bits/stdc++.h>
#define int             long long int
#define pb              push_back
#define ps(x,y)         fixed<<setprecision(y)<<x
#define mod             1000000007
#define w(x)            int x; cin>>x; while(x--)
using namespace std;


vector<int> inTime;
vector<int> outTime;
int timer = 1;

void resize(int n){
    inTime.resize(n+1);
    outTime.resize(n+1);
}

void dfs(int src,int par,vector<int> g[]){
    inTime[src] = timer++;
    for(auto x:g[src]){
        if(x!=par){
            dfs(x,src,g);
        }
    }
    outTime[src] = timer++;
}

bool check(int x,int y){
    if(inTime[x]>inTime[y] and outTime[x]<outTime[y])
        return true;
    return false;
}


int32_t main() {

    int n;
    cin>>n;
    timer = 1;
    resize(n);
    vector<int> g[n+1];
    for(int i=0;i<n-1;i++){
        int x,y;
        cin>>x>>y;
        g[x].push_back(y);
        g[y].push_back(x);
    }
    dfs(1,0,g);
    int q;
    cin>>q;
    for(int i=0;i<q;i++){
        int type,x,y;
        cin>>type>>x>>y;
        if(!check(x,y) and !check(y,x)){
            cout<<"NO\n";
            continue;
        }
        if(type==0){
            if(check(y,x))
                cout<<"YES\n";
            else
                cout<<"NO\n";
        }
        else if(type==1){
            if(check(x,y))
                cout<<"YES\n";
            else
                cout<<"NO\n";
        }

    }
    return 0;
}
