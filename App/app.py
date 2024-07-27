from flask import Flask, render_template, request, jsonify

"""
Note: pip install flask
"""
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('query', '')
    
    print(query)
    top_images =  # these would be changed to the top 4 images with strucutre {id, url}
    return jsonify({'images': top_images})

@app.route('/rank', methods=['POST'])
def rank():
    image_id = request.args.get('image_id', '')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)




