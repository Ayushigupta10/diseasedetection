from pathlib import Path
from flask import Flask,render_template,request,jsonify
from PIL import Image,ImageOps,UnidentifiedImageError
from predictor import Predictor
from wiki import crop_summary,disease_summary
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=8*1024*1024; predictor=Predictor()
@app.get('/')
def index(): return render_template('index.html',models_ready=predictor.ready)
@app.post('/api/analyze')
def analyze():
 f=request.files.get('image')
 if not f:return jsonify(error='Choose an image first.'),400
 if Path(f.filename or '').suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:return jsonify(error='Use a JPG, JPEG, PNG or WEBP image.'),400
 try:
  image=Image.open(f.stream); image=ImageOps.exif_transpose(image); image.load(); image=image.convert('RGB')
 except (UnidentifiedImageError,OSError,ValueError):return jsonify(error='This image cannot be opened. Please choose another file.'),400
 try:result=predictor.predict(image)
 except RuntimeError as exc:return jsonify(error=str(exc)),503
 if result.get('kind')=='disease':result['description']=disease_description(result['crop'],result['disease'])
 elif result.get('kind')=='healthy':result['description']='Your uploaded leaf appears healthy based on the AI model analysis.'
 return jsonify(result)
@app.get('/api/wiki/crop')
def wiki_crop():return jsonify(crop_summary(request.args.get('crop','')))
@app.get('/api/wiki/disease')
def wiki_disease():return jsonify(disease_summary(request.args.get('crop',''),request.args.get('disease','')))

def disease_description(crop,disease):
 descriptions={
 ('Banana','Cordana'):'Cordana leaf spot is a fungal disease associated with Cordana species. It causes brown to dark lesions, often with yellow margins, that can merge and reduce healthy leaf area.',
 ('Banana','Pestalotiopsis'):'Pestalotiopsis leaf spot is a fungal disease. It can produce pale or brown necrotic patches that enlarge and may coalesce, reducing functional leaf tissue.',
 ('Banana','Sigatoka'):'Sigatoka is a fungal leaf disease of banana. It commonly begins as pale streaks that develop into dark elongated lesions and can reduce photosynthesis.',
 ('Corn','Cercospora Leaf Spot Gray Leaf Spot'):'Gray leaf spot is a fungal disease caused by Cercospora species. Long, rectangular tan-to-gray lesions develop between leaf veins and reduce photosynthetic area.',
 ('Corn','Gray Leaf Spot'):'Gray leaf spot is a fungal disease caused by Cercospora species. Long, rectangular tan-to-gray lesions develop between leaf veins and reduce photosynthetic area.',
 ('Corn','Common Rust'):'Common rust is a fungal disease that forms reddish-brown pustules on corn leaves. Severe infection reduces healthy leaf area and can limit grain filling.',
 ('Corn','Northern Leaf Blight'):'Northern corn leaf blight is a fungal disease caused by Exserohilum turcicum. It produces long, cigar-shaped gray-green lesions that can expand across leaves.',
 ('Mango','Anthracnose'):'Mango anthracnose is a fungal disease caused by Colletotrichum species. It can cause dark, sunken spots on leaves, flowers and fruit, especially in humid conditions.',
 ('Mango','Bacterial Canker'):'Mango bacterial canker is associated with Xanthomonas bacteria. It can cause raised dark lesions on leaves and fruit, sometimes with cracking or gum-like exudate.',
 ('Mango','Cutting Weevil'):'Cutting weevil is insect damage rather than an infectious disease. Adult feeding and egg-laying can injure young shoots or leaf stalks and weaken affected growth.',
 ('Mango','Die Back'):'Mango dieback describes progressive drying of shoots and branches, often linked to fungal infection entering through wounds. Symptoms can move from twig tips toward larger branches.',
 ('Mango','Gall Midge'):'Gall midge damage is caused by tiny fly larvae feeding in plant tissue. Infested mango leaves may develop blister-like galls, distortion or premature drying.',
 ('Mango','Powdery Mildew'):'Mango powdery mildew is a fungal disease caused by Oidium species. White powdery growth may cover flowers and young tissues, causing flower loss and reduced fruit set.',
 ('Mango','Sooty Mould'):'Sooty mould is dark fungal growth on honeydew left by sap-feeding insects. The coating can shade leaves; controlling the insect source addresses the underlying cause.',
 ('Millet','Blast'):'Millet blast is a fungal disease caused by Magnaporthe/Pyricularia species. It can form spindle-shaped leaf lesions and may also affect necks or panicles.',
 ('Millet','Rust'):'Millet rust is a fungal disease that produces orange-brown pustules on leaves. Heavy infection reduces green leaf area and can impair plant growth.',
 ('Pomegranate','Alternaria'):'Alternaria disease is a fungal leaf and fruit spot problem. Dark, expanding lesions may cause leaf yellowing or drop and can blemish fruit.',
 ('Pomegranate','Anthracnose'):'Pomegranate anthracnose is a fungal disease associated with Colletotrichum species. It can cause dark, sunken lesions on leaves or fruit, particularly in wet weather.',
 ('Rice','Bacterial Leaf Blight'):'Rice bacterial leaf blight is caused by Xanthomonas oryzae pv. oryzae. Water-soaked leaf-edge lesions expand, turn yellow or straw-colored, and may dry out leaves.',
 ('Rice','Brown Spot'):'Rice brown spot is a fungal disease commonly associated with Bipolaris oryzae. It creates brown oval lesions on leaves and can reduce plant vigor when severe.',
 ('Rice','Leaf Smut'):'Rice leaf smut is a fungal disease associated with Entyloma oryzae. It produces small dark elongated lesions on leaves and can reduce healthy leaf area.',
 ('Soybean','Bacterial Pustule'):'Bacterial pustule is caused by Xanthomonas bacteria. Small raised pustules may appear within yellowish leaf spots, especially on the lower leaf surface.',
 ('Soybean','Frogeye Leaf Spot'):'Frogeye leaf spot is a fungal disease caused by Cercospora sojina. It forms circular lesions with gray centers and dark reddish margins on soybean leaves.',
 ('Soybean','Rust'):'Soybean rust is caused by Phakopsora pachyrhizi. Tan-to-reddish pustules develop on leaves and severe infection can cause early defoliation.',
 ('Soybean','Sudden Death Syndrome'):'Sudden death syndrome is a root disease associated with Fusarium virguliforme. Root infection can lead to interveinal yellowing and necrosis on leaves.',
 ('Soybean','Target Leaf Spot'):'Target spot is a fungal disease associated with Corynespora cassiicola. Brown lesions with concentric rings can enlarge and cause leaf yellowing or drop.',
 ('Soybean','Yellow Mosaic'):'Yellow mosaic is a viral disease complex transmitted by whiteflies in susceptible regions. Leaves show yellow mosaic patterns, mottling or distortion.',
 ('Sugarcane','Mosaic'):'Sugarcane mosaic is a viral disease that causes irregular light and dark green striping or mottling on leaves and may reduce growth.',
 ('Sugarcane','Redrot'):'Sugarcane red rot is a fungal stalk disease caused by Colletotrichum falcatum. Internal stalk tissue becomes red with characteristic pale cross-bands and may decay.',
 ('Sugarcane','Rust'):'Sugarcane rust is a fungal leaf disease that forms orange-brown pustules. Severe infection reduces green leaf area and can limit cane growth.',
 ('Tomato','Bacterial Spot'):'Tomato bacterial spot is caused by Xanthomonas bacteria. Small water-soaked lesions become dark and may cause leaf yellowing, defoliation or fruit spotting.',
 ('Tomato','Early Blight'):'Tomato early blight is associated with Alternaria solani. Dark target-like spots often start on older leaves and may lead to yellowing and defoliation.',
 ('Tomato','Late Blight'):'Tomato late blight is caused by Phytophthora infestans. It can produce rapidly enlarging dark lesions and severe leaf and fruit damage in cool wet conditions.',
 ('Tomato','Leaf Mold'):'Tomato leaf mold is caused by Passalora fulva. Pale yellow patches appear above leaves, with olive-green to brown growth often forming underneath in humid conditions.',
 ('Tomato','Septoria Leaf Spot'):'Septoria leaf spot is caused by Septoria lycopersici. Numerous small circular spots with pale centers and dark margins can yellow and drop lower leaves.',
 ('Tomato','Target Spot'):'Tomato target spot is associated with Corynespora cassiicola. Brown lesions with concentric rings can spread across leaves and cause defoliation.',
 ('Tomato','Tomato Mosaic Virus'):'Tomato mosaic virus causes mottled light and dark green leaves, sometimes with distortion or reduced growth. It can spread through contaminated hands, tools and plant material.',
 ('Tomato','Tomato Yellow Leaf Curl Virus'):'Tomato yellow leaf curl is transmitted by whiteflies. Infected plants can show upward leaf curl, yellowing, stunting and reduced fruit set.',
 ('Tomato','Spider Mites Two Spotted Spider Mite'):'Two-spotted spider mites are pests, not a disease. Feeding causes leaf stippling and bronzing; heavy infestations can cause webbing and leaf drop.'}
 descriptions[('Corn','Blight')]='“Blight” is a broad dataset label and does not identify a specific pathogen. Blight-like symptoms involve rapid leaf tissue browning or death; a crop specialist should confirm the cause.'
 descriptions[('Sugarcane','Yellow')]='“Yellow” is an ambiguous dataset label, not a specific diagnosis. Yellowing can have several infectious and environmental causes, so this image label alone cannot support a verified cause or treatment.'
 return descriptions.get((crop,disease),f'No verified description is available for the model label “{disease}” on {crop}. This label is an image-model prediction, not a confirmed diagnosis. Check symptoms with a local agricultural extension specialist before treating the plant.')
if __name__=='__main__':app.run(debug=False)
