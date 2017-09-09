import tensorflow as tf

def main():

    x = tf.constant([35, 40, 45], name="x")
    y = tf.Variable(x + 5, name="y")

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        print(session.run(y))

if __name__ == "__main__":
    main()
