function e=bandlp(z,dindex,pindex)
y=z(:,pindex);
X=z(:,dindex);
b=pinv(X'*X)*X'*y;
y_hat=X*b;
r=y-y_hat;
e=norm(r);

