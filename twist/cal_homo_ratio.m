function homo_ratio = cal_homo_ratio(adj, label, self_loop)
    %class_num = max(label) + 1;
    y_onehot = full(ind2vec(label'))';
    adj_y = y_onehot * y_onehot';

    if self_loop
        adj = adj - eye(size(adj, 1));
    end

    homo = sum(sum(adj_y .* adj));
    homo_ratio = homo / sum(sum(adj));
end