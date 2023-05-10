tic
usercount=max(unique(rating(:,1)));
itemcount=max(unique(rating(:,2)));

lll=randperm(size(rating,1));
traincount=ceil(size(rating,1)*0.2);
%  traincount=ceil(size(rating,1)*0.7);
% traincount=ceil(size(rating,1)*0.6);

testcount=size(rating,1)-traincount;
train_rating=rating(lll(1:traincount),:);
test_rating=rating(lll(traincount+1:size(rating,1)),:);

%% 
usercount=max(unique(ratings(:,1)));
itemcount=max(unique(ratings(:,2)));

lll=randperm(size(ratings,1));
traincount=ceil(size(ratings,1)*0.6);
%  traincount=ceil(size(rating,1)*0.7);
% traincount=ceil(size(rating,1)*0.6);

testcount=size(ratings,1)-traincount;
train_ratings=ratings(lll(1:traincount),:);
test_ratings=ratings(lll(traincount+1:size(ratings,1)),:);
test_rating = test_ratings; train_rating =train_ratings;

UI_train_matrix=zeros(usercount,itemcount);
for i=1:size(train_rating,1)   
   UI_train_matrix(train_rating(i,1),train_rating(i,2))=train_rating(i,3);
end
UI_train_matrix=sparse(usercount,itemcount);
for i=1:size(train_rating,1)   
   UI_train_matrix(train_rating(i,1),train_rating(i,2))=train_rating(i,3);
end



userrating=zeros(usercount,5338);
for i=1:usercount
   currentindex=find(UI_matrix(i,:)>0);
   userrating(i,1)=length(currentindex);
   userrating(i,2:userrating(i,1)+1)=currentindex; 
end


UI_test_matrix=zeros(usercount,itemcount);
for i=1:size(test_rating,1)   
   UI_test_matrix(test_rating(i,1),test_rating(i,2))=test_rating(i,3);
end


Ui_PRE = zeros(usercount,itemcount);
for i=1:size(test_rating,1)   
   Ui_PRE(test_rating(i,1),test_rating(i,2))=pre(i,1);
end


userrating=zeros(usercount,5338);
for i=1:usercount
   currentindex=find(UI_train_matrix(i,:)>0);
   userrating(i,1)=length(currentindex);
   userrating(i,2:userrating(i,1)+1)=currentindex;
    
end

network=zeros(usercount,20);
trustcount=size(trustnetwork,1);
for i=1:trustcount
   currentindex=trustnetwork(i,1);
   network(currentindex,1)=network(currentindex,1)+1;
   network(currentindex,network(currentindex,1)+1)=trustnetwork(i,2);
    
end

trustMatrix=zeros(usercount,usercount);
for i=1:size(trust,1)
user_num=trust(i,1);
trust_num=trust(i,2);
trustMatrix(user_num,trust_num)=1;
end
toc