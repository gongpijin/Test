import tensorflow as tf
x_train=[1,2,3]
y_train=[1,2,3]
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
#构建拟合函数
h=x_train*w+b
#构建损失函数
cost=tf.reduce_mean(tf.square(h-y_train))
#采用梯度下降更新权重
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
#运算计算图执行训练
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    if step % 20==0:
        print(step,sess.run(cost),sess.run(w),sess.run(b))
print(sess.run(w))
print(sess.run(b))
print(sess.run(cost))
print("aaaaa")






