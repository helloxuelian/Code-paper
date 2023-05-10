%  负采样用来测试
neg_num_test = 100;
 user_unique = unique(test_rating(:,1));
 test_100 = zeros(length(user_unique)*neg_num_test,3);
for user_uniqueid=1:length(user_unique)
    userid = user_unique(user_uniqueid,1);
   idx = find(UI_test_matrix(userid,:)==0&(UI_train_matrix(userid,:)==0));
    test_100(1+(user_uniqueid-1)*neg_num_test:user_uniqueid*neg_num_test,1) = userid;
    test_100(1+(user_uniqueid-1)*neg_num_test:user_uniqueid*neg_num_test,2) =randsample(idx,neg_num_test)';
    test_100(1+(user_uniqueid-1)*neg_num_test:user_uniqueid*neg_num_test,3) = 0;
end
test_final = [test_rating;test_100];

% 负采样用来训练
neg_num = 1;
 test_1 = zeros(length(train_rating)*neg_num,3);
for user_uniqueid=1:length(train_rating)
    userid = train_rating(user_uniqueid,1);
   idx = find(UI_test_matrix(userid,:)==0&(UI_train_matrix(userid,:)==0));
    test_1(1+(user_uniqueid-1)*neg_num:user_uniqueid*neg_num,1) = userid;
    test_1(1+(user_uniqueid-1)*neg_num:user_uniqueid*neg_num,2) =randsample(idx,neg_num)';
    test_1(1+(user_uniqueid-1)*neg_num:user_uniqueid*neg_num,3) = 0;
end
train_rating = [train_rating;test_1];



% 负采样2
neg_num = 1;
 user_unique = unique(train_rating(:,1));
  test_1 = zeros(length(train_rating)*neg_num,3);
current_id_test = 1;
for user_uniqueid=1:length(user_unique)
    userid = user_unique(user_uniqueid,1);
    neg_samp_num_train = userrating(userid,1);
   idx = find(UI_test_matrix(userid,:)==0&(UI_train_matrix(userid,:)==0));
    test_1(current_id_test:current_id_test+neg_samp_num_train-1,1) = userid;
    test_1(current_id_test:current_id_test+neg_samp_num_train-1,2) =randsample(idx,neg_samp_num_train)';
    test_1(current_id_test:current_id_test+neg_samp_num_train-1,3) = 0;
        current_id_test = current_id_test+neg_samp_num_train;

end
train_rating = [train_rating;test_1];
