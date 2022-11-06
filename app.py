from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    a = float(request.form.get('a'))
    b = float(request.form.get('b'))
    c = float(request.form.get('c'))
    d = float(request.form.get('d'))
    e = str(request.form.get('e'))
    f = float(request.form.get('f'))
    g = float(request.form.get('g'))
    h = float(request.form.get('h'))
    i = float(request.form.get('i'))
    j = float(request.form.get('j'))
    k = float(request.form.get('k'))
    l = float(request.form.get('l'))
    m = float(request.form.get('m'))
    n = float(request.form.get('n'))
    o = float(request.form.get('o'))
    p = float(request.form.get('p'))
    q = float(request.form.get('q'))
    r = float(request.form.get('r'))
    s = float(request.form.get('s'))
    t = float(request.form.get('t'))
    u = float(request.form.get('u'))
    v = float(request.form.get('v'))
    w = float(request.form.get('w'))
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    z = float(request.form.get('z'))
    aa = float(request.form.get('aa'))
    bb = float(request.form.get('bb'))
    cc = float(request.form.get('cc'))
    dd = float(request.form.get('dd'))
    ee = float(request.form.get('ee'))
    ff = float(request.form.get('ff'))
    gg = float(request.form.get('gg'))
    hh = float(request.form.get('hh'))

    # prediction
    result = model.predict(np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
                                     q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee, ff, gg, hh]).reshape(1, 34))
    print(str(result))
    return str(result)


if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', debug=False, port=environ.get("PORT", 5000))


