"""Wikipedia API enrichment with explicit crop URLs and strict disease-title matching."""
import re,requests
from urllib.parse import quote_plus
API='https://en.wikipedia.org/w/api.php'; HEADERS={'User-Agent':'SAgriVision/1.0 (plant image classifier)'}
CROP_ARTICLES={'Banana':('Banana','https://en.wikipedia.org/wiki/Banana'),'Corn':('Maize','https://en.wikipedia.org/wiki/Maize'),'Mango':('Mango','https://en.wikipedia.org/wiki/Mango'),'Millet':('Millet','https://en.wikipedia.org/wiki/Millet'),'Pomegranate':('Pomegranate','https://en.wikipedia.org/wiki/Pomegranate'),'Rice':('Rice','https://en.wikipedia.org/wiki/Rice'),'Soybean':('Soybean','https://en.wikipedia.org/wiki/Soybean'),'Sugarcane':('Sugarcane','https://en.wikipedia.org/wiki/Sugarcane'),'Tomato':('Tomato','https://en.wikipedia.org/wiki/Tomato')}
def _pages(params):
 r=requests.get(API,params={**params,'action':'query','format':'json','formatversion':2},headers=HEADERS,timeout=5); r.raise_for_status(); return r.json().get('query',{}).get('pages',[])
def _details(page,url=None): return {'available':True,'title':page.get('title',''),'summary':page.get('extract','')[:1400],'image':page.get('thumbnail',{}).get('source'),'url':url or page.get('fullurl')}
def crop_summary(crop):
 item=CROP_ARTICLES.get(crop)
 if not item:return {'available':False,'message':'Crop article unavailable.'}
 title,url=item
 try:
  page=next((p for p in _pages({'titles':title,'redirects':1,'prop':'extracts|pageimages|info','exintro':1,'explaintext':1,'pithumbsize':640,'inprop':'url'}) if not p.get('missing')),None)
  return _details(page,url) if page else {'available':False,'message':'Wikipedia summary is unavailable.','title':title,'url':url}
 except Exception:return {'available':False,'message':'Wikipedia information is temporarily unavailable. The crop article link is still available.','title':title,'url':url}
def _tokens(text):return set(re.findall(r'[a-z0-9]+',text.lower()))
def disease_summary(crop,disease):
 if crop not in CROP_ARTICLES or not disease:return {'available':False,'message':'No matching disease article was found.'}
 search_url='https://en.wikipedia.org/w/index.php?search='+quote_plus(f'{crop} {disease} plant disease')
 try:
  pages=_pages({'generator':'search','gsrsearch':f'"{disease}" {crop} plant disease','gsrlimit':8,'prop':'extracts|pageimages|info','exintro':1,'explaintext':1,'pithumbsize':640,'inprop':'url'})
  required=_tokens(disease)-{'leaf','plant','disease','two','spotted'}
  matches=[p for p in pages if required and required.issubset(_tokens(p.get('title','')))]
  if not matches:return {'available':False,'message':f'Wikipedia has no dedicated article matching “{disease}” on {crop}. You can search related Wikipedia pages below; the model label is not a confirmed diagnosis.','url':search_url,'link_label':'Search Wikipedia for this disease'}
  matches.sort(key=lambda p:(crop.lower() not in p.get('title','').lower(),len(p.get('title',''))))
  return _details(matches[0])
 except Exception:return {'available':False,'message':'Wikipedia could not be reached right now. Search for related disease pages below.','url':search_url,'link_label':'Search Wikipedia for this disease'}
