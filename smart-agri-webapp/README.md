# واجهة محلية لتصنيف الأراضي (Smart Agriculture — Milestone 3)

واجهة ويب بسيطة تعمل **محليًا فقط على جهازك** (لا رفع لأي Cloud). ترفع صورة قمر صناعي
(على نمط EuroSAT) وتُخبرك النموذج المدرَّب في الـ notebook بالتصنيف التفصيلي (10 فئات)
وبفئة الكتيّب (Agriculture / Water / Urban / Roads / Trees) المطابقة لِـ Part F من الـ notebook.

## 1) شغّل الـ Notebook أولاً حتى آخر خلية

شغّل `Smart_Agriculture_M3_Optimization.ipynb` كاملاً (على Colab بـ GPU، أو محليًا إن
توفر عندك GPU/وقت كافٍ). آخر خلايا في الـ notebook (قسم "Deliverables & summary") تحفظ:

```
outputs/models/eurosat_resnet_final.pt
outputs/models/model_config.json
```

## 2) انسخ ملفي النموذج إلى هذا المشروع

انسخ الملفين السابقين إلى مجلد `model/` هنا، بحيث يصبح الشكل:

```
smart-agri-webapp/
  model/
    eurosat_resnet_final.pt
    model_config.json
  app.py
  model_def.py
  ...
```

## 3) ثبّت المتطلبات (مرة واحدة)

```bash
cd smart-agri-webapp
python -m venv .venv
source .venv/bin/activate        # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt
```

> ملاحظة: `torch`/`torchvision` بدون GPU تكفي تمامًا للـ inference (تصنيف صورة واحدة)
> سرعتها ممتازة على CPU، فلا داعي لإعداد CUDA على جهازك المحلي.

## 4) شغّل السيرفر المحلي

```bash
python app.py
```

ثم افتح المتصفح على:

```
http://127.0.0.1:5000
```

ارفع صورة واضغط "تصنيف الصورة". كل شيء — رفع الصورة، التحليل، والنتيجة — يحدث على جهازك
فقط عبر `localhost`؛ لا يوجد أي اتصال بالإنترنت أو بأي خدمة سحابية.

## كيف تعمل الواجهة

- `app.py`: خادم Flask صغير — صفحة رفع (`/`) ونقطة نهاية تصنيف (`/predict`).
- `model_def.py`: نفس تعريف الشبكة (`build_model` / `make_head`) ونفس خطوات تجهيز الصورة
  (`preprocess_image`) الموجودة في الـ notebook، حتى تُحمَّل الأوزان المحفوظة بشكل صحيح
  وتُعطي نفس نتائج الـ notebook تمامًا. كما يحتوي على خريطة الفئات العشر → فئات الكتيّب
  الخمس (Agriculture/Water/Urban/Roads/Trees) المطابقة لِـ Part F.
- `templates/index.html` + `static/style.css`: واجهة رفع بالسحب والإفلات (drag & drop)
  بالعربية، تعرض الفئة التفصيلية + فئة الكتيّب + نسبة الثقة + أفضل احتمالين آخرين.

## إذا ظهرت رسالة "لم يتم العثور على ملفات النموذج"

يعني أن `model/eurosat_resnet_final.pt` أو `model/model_config.json` غير موجودين بعد —
ارجع للخطوة 1 و2.

## ملاحظة عن موضوع المشروع

ملف الـ notebook الذي أرسلته (EuroSAT — تصنيف صور أقمار صناعية إلى 10 فئات أرض، مُجمّعة
في Agriculture/Water/Urban/Roads/Trees) يختلف عن موضوع "Crop Production in India" الموصوف
في Project Proposal.pdf (بيانات جدولية عن إنتاج المحاصيل في الهند، بدون صور أقمار صناعية).
هذه الواجهة مبنية حسب ملف الـ notebook الذي طلبت تنفيذه (تصنيف صورة → فئة أرض)، فتأكد أن
هذا هو المطلوب فعلاً من الـ Milestone 3 عندك قبل التسليم.
