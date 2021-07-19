
// LINKED LIST

//1>> Reverse a linked list // TC o(n)
// Iterative
void reverse(){
        Node* current = head;
        Node *prev = NULL, *next = NULL;
        while (current != NULL) {
            next = current->next;
            current->next = prev;
            prev = current;
            current = next;
        }
        head = prev;
    }
// Recursive
    Node* reverse(Node* head){
        if (head == NULL || head->next == NULL)
            return head;
        Node* rest = reverse(head->next);
        head->next->next = head;
        head->next = NULL;
        return rest;
    }

//2>> Reverse a Linked List in groups of given size // TC o(n)
struct node *reverse (struct node *head, int k){
    node* prev=NULL;
    node* curr=head;
    node* q;
    int counter=k;
    while(counter-- && curr!=NULL){
    q=curr->next;
    curr->next=prev;
    prev=curr;
    curr=q;
    }
    if(curr!=NULL) head->next= reverse(q,k);
    return prev;
}

//3>> Detect Loop in linked list // TC o(n)
bool detectLoop(Node* head){
    Node* slow = head;
    Node* fast;
    if(head) fast = head->next;
    while(fast && slow && fast->next){
        slow=slow->next;
        fast=fast->next->next;
        if(slow==fast){
        return true;
    }

//4>> Write a program to Delete loop in a linked list. // TC o(n)
      void removeLoop(Node* head){
          Node *slow=head;
      Node *fast = head;
      if(head == NULL || head->next == NULL) return;
      while(fast!=NULL && fast->next!=NULL){
        slow = slow->next;
        fast = fast->next->next;
      if(slow == fast) break;
      }
      if(slow==head){
        while(slow->next!=head){
        slow=slow->next;
        }
        slow->next = NULL;
      }
      if(slow == fast){
        slow = head;
        while(slow->next != fast->next){
            if(slow == fast->next){
            fast->next == NULL;
            }
            slow = slow->next;
            fast = fast->next;
        }
        fast->next=NULL;
    }
}

//5>> Find first node of loop in a linked list // TC o(n)
Node* detectAndRemoveLoop(Node* head){
    if (head == NULL || head->next == NULL) return NULL;
    Node *slow = head, *fast = head;
    slow = slow->next;
    fast = fast->next->next;
    while (fast && fast->next) {
        if (slow == fast) break;
        slow = slow->next;
        fast = fast->next->next;
    }
    if (slow != fast) return NULL;
    slow = head;
    while (slow != fast){
        slow = slow->next;
        fast = fast->next;
    }
    return slow;
}

//6>> Remove duplicate element from sorted Linked List // TC o(n)
Node *removeDuplicates(Node *root){
 Node* prev = NULL;
 Node* curr = root;
 while(curr){
     prev = curr;
     curr = curr->next;
     if(curr){
     if(prev->data == curr->data){
         prev->next = curr->next;
         curr = prev;
     }
     }
 }
 return root;
}

//7>> Remove duplicates from an unsorted linked list // TC o(n) // SC o(n)
Node * removeDuplicates( Node *head){
 unordered_set<int> seen;
 Node* curr = head;
 Node* prev = NULL;
 while(curr){
     if(seen.find(curr->data) != seen.end()){
        prev->next = curr->next;
        curr = prev->next;
     }
     else{
         seen.insert(curr->data);
         prev = curr;
         curr = curr->next;
     }
 }
 return head;
}

//8>> Move last element to front of a given Linked List
void moveToFront(Node **head_ref)  {
    if (*head_ref == NULL || (*head_ref)->next == NULL) return;
    Node *secLast = NULL;
    Node *last = *head_ref;
    while (last->next != NULL){
        secLast = last;
        last = last->next;
    }
    secLast->next = NULL;
    last->next = *head_ref;
    *head_ref = last;
}
void push(Node** head_ref, int new_data)  {
    Node* new_node = new Node();
    new_node->data = new_data;
    new_node->next = (*head_ref);
    (*head_ref) = new_node;
}

//9>> Add 1 to a number represented as linked list // TC o(n)
    Node* addOne(Node *head){
        struct Node *p;
        struct Node *q=head;
    while(q->next!=NULL) q=q->next;
    if(q->data!=9){
      q->data++;
      return head;
    }
    else{
    while(head!=q){
    q->data=0;
      p=head;
      while(p->next!=q){
      p=p->next;
      }
      if(p->data==9) q=p;
      else{
      p->data++;
      return head;
      }
    }
    head->data++;
    return head;
    }
  }

//10>> Add two numbers represented by linked lists // TC o(m+n) //TC o(m+n)
Node* reverseList(Node* head){
  Node* current = head;
  Node* prev = NULL;
  Node* next = NULL;
  while(current != NULL){
  next = current -> next;
  current -> next = prev;
  prev = current;
  current = next;
}
 return prev;
}
Node* addTwoLists(Node* first, Node* second){
first = reverseList(first);
second = reverseList(second);
int carry = 0, sum = 0;
Node* start = NULL;
Node* end = NULL;
while(first != NULL || second != NULL){
    int a = (first != NULL)?first -> data:0;
    int b = (second != NULL)?second -> data:0;
    sum = carry + (a + b);
    carry = (sum >= 10)?1:0;
    sum = sum % 10;
if(start == NULL){
  start = new Node(sum);
  end = start;
}
else{
  end->next = new Node(sum);
  end = end->next;
}
if(first != NULL) first = first->next;
if(second != NULL) second = second->next;
}
if(carry > 0) end->next = new Node(carry);
start = reverseList(start);
return start;
}

//11>>Intersection of two sorted Linked lists // TC o(m+n) // SC o(m+n)
Node* findIntersection(Node* head1, Node* head2){
   Node* newhead = NULL;
   Node* tr;
    while(head1 && head2){
        if(head1->data == head2->data){
            if(!newhead){
            newhead = new Node(head1->data);
            tr = newhead;
            head1=head1->next;
            head2=head2->next;
            continue;
            }
            tr->next = new Node(head1->data);
            tr = tr->next;
            head1=head1->next;
            head2=head2->next;
        }
        else if(head1->data < head2->data) head1=head1->next;
        else head2=head2->next;
    }
    if(tr) tr->next=NULL;
    return newhead;
}

//12>> Intersection Point in Y Shapped Linked Lists // TC o(n+m)
int intersectPoint(Node* head1, Node* head2){
    Node* l1 = head1;
    Node* l2 = head2;
    while(l1!=l2){
        if(l1) l1=l1->next;
        else l1 = head2;
        if(l2) l2=l2->next;
        else l2 = head1;
    }
    return l1->data;
}

//13>> Find the middle Element of a linked list // TC o(n)
ListNode* middleNode(ListNode* head){
        if(!head && !head->next) return head;
        ListNode* slow = head;
        ListNode* fast = head->next;
    while(fast && fast->next){
        slow = slow->next;
        fast = fast->next->next;
    }
       if(fast) return slow->next;
        return slow;
  }

//14>> Check If Circular Linked List // TC o(n)
bool isCircular(Node *head){
    if(!head) return true;
   Node *t = head;
   while(t && t->next!=head){
       t=t->next;
   }
   if(!t) return false;
   else return true;
}

//15>> Split a Circular Linked List into two halves // TC o(n)
void splitList(Node *head, Node **head1_ref, Node **head2_ref){
    Node *slow = head, *fast = head;
    while(fast->next != head && fast->next->next != head){
        slow = slow->next;
        fast = fast->next->next;
    }
    *head1_ref = head;
    *head2_ref = slow->next;
    Node *temp = slow->next;
    slow->next = head;
    Node *curr = *head2_ref;
    while(curr->next != head){
        curr=curr->next;
    }
    curr->next = *head2_ref;
}

//16>> Check if Linked List is Palindrome // TC o(n)
bool isPalindrome(Node* head) {
        string str="";
        Node* t=head;
        while(t!=NULL){
            str.push_back(t->data);
            t=t->next;
        }
        return checkit(str);
}

//17>> Deletion from a Circular Linked List // TC o(o)
void deleteNode(Node** head, int key) {
    if (*head == NULL) return;
    if((*head)->data==key && (*head)->next==*head){
        free(*head);
        *head=NULL;
        return;
    }
    Node *last=*head,*d;
    if((*head)->data==key) {
        while(last->next!=*head)
            last=last->next;
        last->next=(*head)->next;
        free(*head);
        *head=last->next;
    }
    while(last->next!=*head&&last->next->data!=key){
        last=last->next;
    }
    if(last->next->data==key){
        d=last->next;
        last->next=d->next;
        free(d);
    }
    else
        cout<<"no such keyfound";
    }

//18>> Reverse a Doubly Linked List // TC o(n)
Node* reverseDLL(Node * head){
  Node* p=head;
  Node* q=head;

  while(q){
    p=q;
    Node* temp=q->prev;
    q->prev=q->next;
    q->next=temp;
    q=q->prev;
  }
  return p;
}

//19>> Find pairs with given sum in doubly linked list // TC o(n)
void pairSum(struct Node *head, int x){
    struct Node *first = head;
    struct Node *second = head;
    while (second->next != NULL)
        second = second->next;
    bool found = false;
    while (first != NULL && second != NULL &&
        first != second && second->next != first){
        if ((first->data + second->data) == x){
            found = true;
            cout << "(" << first->data<< ", "
                << second->data << ")" << endl;
            first = first->next;
            second = second->prev;
        }
        else {
            if ((first->data + second->data) < x) first = first->next;
            else second = second->prev;
        }
    }
    if (found == false)
        cout << "No pair found";
}

//20>> Count triplets in a sorted doubly linked list whose sum is equal
// to a given value x // TC o(nn)
int countPairs(struct Node* first, struct Node* second, int value){
    int count = 0;
    while (first != NULL && second != NULL &&
           first != second && second->next != first) {
        if ((first->data + second->data) == value) {
            count++;
            first = first->next;
            second = second->prev;
        }
        else if ((first->data + second->data) > value)
            second = second->prev;
        else first = first->next;
    }
    return count;
}
int countTriplets(struct Node* head, int x){
    if (head == NULL) return 0;
    struct Node* current, *first, *last;
    int count = 0;
    last = head;
    while (last->next != NULL)
        last = last->next;
    for (current = head; current != NULL; current = current->next) {
        first = current->next;
        count += countPairs(first, last, x - current->data);
    }
    return count;
}

//21>> Sort a k sorted doubly linked list // TC o(nlogk)
struct compare {
    bool operator()(struct Node* p1, struct Node* p2){
        return p1->data > p2->data;
    }
};
struct Node* sortAKSortedDLL(struct Node* head, int k){
    if (head == NULL) return head;
    priority_queue<Node*, vector<Node*>, compare> pq;
    struct Node* newHead = NULL, *last;
    for (int i = 0; head != NULL && i <= k; i++) {
        pq.push(head);
        head = head->next;
    }
    while (!pq.empty()){
        if (newHead == NULL) {
            newHead = pq.top();
            newHead->prev = NULL;
            last = newHead;
        }
        else {
            last->next = pq.top();
            pq.top()->prev = last;
            last = pq.top();
        }
        pq.pop();
        if (head != NULL) {
            pq.push(head);
            head = head->next;
        }
    }
    last->next = NULL;
    return newHead;
}
void push(struct Node** head_ref, int new_data){
    struct Node* new_node =
          (struct Node*)malloc(sizeof(struct Node));
    new_node->data = new_data;
    new_node->prev = NULL;
    new_node->next = (*head_ref);
    if ((*head_ref) != NULL)
        (*head_ref)->prev = new_node;
    (*head_ref) = new_node;
}

//22>> Flattening a Linked List // TC o(mn)
Node * merge(Node* a, Node* b){
    Node * temp = new Node(0);
    Node * res = temp;
    while(a && b){
        if(a->data<b->data){
            temp->bottom = a;
            temp = temp->bottom;
            a = a->bottom;
        }
        else{
            temp->bottom = b;
            temp = temp->bottom;
            b= b->bottom;
        }
    }
    if(a) temp->bottom = a;
    else temp->bottom = b;
    return res->bottom;
}
Node *flatten(Node *root){
   if(root==NULL || root->next==NULL) return root;
   root->next=flatten(root->next);
   root=merge(root,root->next);
   return root;
}

//23>> Given a linked list of 0s, 1s and 2s, sort it // TC o(n)
Node* segregate(Node *head){
  int arr[3]={0};
  Node*temp=head;
  while(temp!=NULL){
    arr[temp->data]++;
    temp=temp->next;
  }
  temp=head;
  for(int i=0;i<3;i++){
    while(arr[i]--){
      temp->data=i;
      temp=temp->next;
      }
  }
  return head;
}

//24>> Clone a linked list with next and random pointer // TC o(n)
Node *copyList(Node *head) {
      Node *iter = head;
      Node *front = head;
      while (iter != NULL) {
        front = iter->next;
        Node *copy = new Node(iter->data);
        iter->next = copy;
        copy->next = front;
        iter = front;
      }
      iter = head;
      while (iter != NULL) {
        if (iter->arb != NULL) {
          iter->next->arb = iter->arb->next;
        }
        iter = iter->next->next;
      }
      iter = head;
      Node *pseudoHead = new Node(0);
      Node *copy = pseudoHead;
      while (iter != NULL) {
        front = iter->next->next;
        copy->next = iter->next;
        iter->next = front;
        copy = copy -> next;
        iter = front;
      }
      return pseudoHead->next;
}

//25>> Merge K sorted linked lists // o(nklogk) // o(k)
class MyComparator{
    public:
        int operator()(Node *a, Node *b){
            return b->data<a->data;
        }
};
Node * mergeKLists(Node *arr[], int N){
    priority_queue<Node *, vector<Node *>, MyComparator> q;
     for(int i=0;i<N;i++){
        if(arr[i]) q.push(arr[i]);
     }
     Node *dummy = new Node(0);
     Node *tail = dummy;
     while(!q.empty()){
         Node *temp = q.top();
         q.pop();
         if(temp->next) q.push(temp->next);
         tail->next = temp;
         tail = tail->next;
     }
     return dummy->next;
}

//26>> Multiply two linked lists // TC o(n)
long long  multiplyTwoLists (Node* l1, Node* l2){
  long long int num1 = 0; long long int num2= 0;
    while (l1 || l2){
    if (l1){
    num1 =( num1*10 + l1->data)%1000000007;
    l1 = l1->next;
    }
    if (l2){
    num2 = (num2*10 + l2->data)%1000000007;
    l2 = l2->next;
    }
   }
    return (num1*num2)%1000000007;
}

//27>> Delete nodes having greater value on right // TC o(n)
Node* Reverse(Node* head) {
    if (!head || !head->next) return head;
    Node* prev = nullptr;
    Node* cur = head;
    Node* next = nullptr;
    while (cur) {
      next = cur->next;
      cur->next = prev;
      prev = cur;
      cur = next;
    }
    return prev;
}

Node *compute(Node *head){
    head = Reverse(head);
    if(!head && !head->next) return head;
    Node *temp = head;
    int max = temp->data;
    while(temp->next){
        if(temp->next->data>=max){
            max=temp->next->data;
            temp=temp->next;
        }
        else{
            temp->next=temp->next->next;
        }
    }
    head = Reverse(head);
    return head;
}

//28>> Segregate even and odd nodes in a Link List // TC o(n) // SC o(n)
Node* segregate(Node* head){
    Node* oddH = new Node(0);
    Node* evenH = new Node(0);
    Node* odd = oddH;
    Node* even = evenH;
    Node* temp = head;
    while(temp){
      if(temp->data%2==0){
        even->next = temp;
        even = even->next;
    }
    else{
      odd->next = temp;
      odd = odd->next;
    }
    temp = temp->next;
    }
    even->next = oddH->next;
    odd->next = NULL;
    head = evenH->next;
    delete(evenH);
    delete(oddH);
    return head;
}

//29>> Nth node from end of linked list // TC O(n)
int getNthFromLast(Node *head, int n){
  Node* temp=head;
  for(int i=1;i<n;i++){
      if(temp->next==NULL)
      return -1;
     temp=temp->next;
  }
  Node* slow=head;
  while(temp->next!=NULL){
    temp=temp->next;
    slow=slow->next;
  }
  return slow->data;
}

//30>> First non-repeating character in a stream // o(26n)
string FirstNonRepeating(string A){
  int arr[26] = {0};
  queue<char> q;
  string res = "";
  for(int i = 0; i < A.length(); i++) {
    arr[A[i] - 'a']++;
    q.push(A[i]);
  while(!q.empty()){
    if(arr[q.front() - 'a'] > 1) q.pop();
    else break;
  }
  if(q.empty()) res += '#';
  else res += q.front();
  }
  return res;
}

//31>> merge sort
Node* merge(Node* l1,Node* l2){
 if(!l1) return l2;
 if(!l2) return l1;

if(l1->data  < l2->data){
   l1->next=merge(l1->next,l2);
   return l1;
}
else{
   l2->next=merge(l1,l2->next);
   return l2;
}
}
Node* mergeSort(Node* head) {
 if(head==NULL || head->next==NULL)
 return head;
 Node* slow=head;
 Node* fast=head->next;
 while(fast && fast->next){
  slow=slow->next;
  fast=fast->next->next;
}
 Node* newHead=slow->next;
 slow->next=NULL;
 return merge(mergeSort(head),mergeSort(newHead));
}

//32>> quick sort
struct node * partition(struct node *head, struct node *tail){
  node * prev=head, *cur=head->next;
  node *pivot = head;
  while(cur != tail->next){
    if(cur->data < pivot->data){
    swap(prev->next->data,cur->data);
    prev = prev->next;
}
  cur = cur->next;
}
  swap(pivot->data,prev->data);
  return prev;
}

void quickSortRec(struct node * head, struct node *tail){
  if(head == tail || tail == NULL || head == NULL) return;
  struct node *pivot = partition(head , tail);
  quickSortRec(head, pivot);
  quickSortRec(pivot->next, tail);
}

void quickSort(struct node **headRef) {
  node *tail = *headRef;
  while(tail->next != NULL)
  tail = tail->next;
  quickSortRec(*headRef, tail);
}

//33>> Rotate Doubly linked list by N nodes // TC o(n)
// This function rotates a doubly ll counter-clockwise and updates the
// head. The function assumes that N is smallerthan size of ll. It
// doesn't modify the list if N is greater than or equal to size
void rotate(struct Node** head_ref, int N) {
    if (N == 0) return;
    struct Node* current = *head_ref;
    int count = 1;
    while (count < N && current != NULL) {
        current = current->next;
        count++;
    }
    if (current == NULL) return;
    struct Node* NthNode = current;
    while (current->next != NULL)
        current = current->next;
    current->next = *head_ref;
    (*head_ref)->prev = current;
    *head_ref = NthNode->next;
    (*head_ref)->prev = NULL;
    NthNode->next = NULL;
}

//34>> Reverse a doubly linked list in groups of given size // TC o(n)
//function to add at begginning
void push(Node** head_ref, Node* new_node) {
    // since we are adding at the beginning,
    new_node->prev = NULL;
    new_node->next = (*head_ref);
    if ((*head_ref) != NULL)
        (*head_ref)->prev = new_node;
    (*head_ref) = new_node;
}
Node* revListInGroupOfGivenSize(Node* head, int k) {
    Node *current = head;
    Node* next = NULL;
    Node* newHead = NULL;
    int count = 0;
    while (current != NULL && count < k) {
        next = current->next;
        push(&newHead, current);
        current = next;
        count++;
    }
    // if next group exists then making the desired
    // adjustments in the link
    if (next != NULL) {
        head->next = revListInGroupOfGivenSize(next, k);
        head->next->prev = head;
    }
    return newHead;
}
