from flask import Flask, render_template, request, jsonify

"""
Note: pip install flask
"""
app = Flask(__name__)


"""
These are just place holder images, can be deleted later
"""
placeholder_images = [
    {'id': '1', 'url': 'https://media.istockphoto.com/id/1672317574/photo/ama-dablam-mountain-peak.webp?b=1&s=170667a&w=0&k=20&c=Ea8yDEHpUemrRuMZUKGPDBE11YTWVksIupMN8FkEBf8='},
    {'id': '2', 'url': 'https://static.vecteezy.com/system/resources/thumbnails/026/542/204/small_2x/landscape-natural-beautiful-mountains-and-blue-sky-panorama-photo.jpg'},
    {'id': '3', 'url': 'https://assets.simpleviewinc.com/simpleview/image/upload/c_fill,g_xy_center,h_400,q_75,w_375,x_3185,y_2249/v1/clients/vancouverusa/Looking_North__80691a30-4073-4a95-a33d-4c3355104034.jpg'}
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    top_images = placeholder_images # these would be changed to the top 4 images with strucutre {id, url}
    return jsonify({'images': top_images})

@app.route('/rank', methods=['POST'])
def rank():
    image_id = request.args.get('image_id', '')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)


