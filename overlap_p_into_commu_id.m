
max_community=10;
user_community =zeros(usercount,max_community);
set_threathold=0.0001;

for  i=1:usercount
    user_community(i,1:length(find(Com_P(i,:)>set_threathold)))=find(Com_P(i,:)>set_threathold);
end