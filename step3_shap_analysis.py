import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def perform_shap_analysis(data_file='processed_maritime_data.csv'):
    # تحميل النموذج والبيانات
    try:
        model = joblib.load('xgboost_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        df = pd.read_csv(data_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please make sure step1 and step2 are completed.")
        return

    # استخراج جزء من البيانات للتفسير (توفيراً للوقت والذاكرة)
    # سنأخذ أحدث 500 عينة كبيانات خلفية لتفسير SHAP
    df = df.sort_values('datetime').reset_index(drop=True)
    X = df[feature_columns]
    X_sample = X.tail(500)

    # 1. تهيئة shap.TreeExplainer للنموذج المدرب
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 2. استخراج مخطط الأهمية الشاملة (Summary Plot)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("مخطط أهمية العوامل الشاملة - Summary Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("تم حفظ مخطط الأهمية في shap_summary_plot.png")
    
    """
    تعليق لمتخذ القرار حول Summary Plot:
    - يوضح هذا المخطط أهمية كل ميزة على حدة وتأثيرها على التنبؤ (عدد الحوادث).
    - المتغيرات في الأعلى هي الأكثر تأثيراً في النموذج (مثل ارتفاع الموج، الرؤية، أو سرعة الرياح).
    - الألوان تعبر عن قيمة المتغير (أحمر = قيمة عالية، أزرق = قيمة منخفضة).
    - اتجاه النقط (يمين أو يسار الصفر) يوضح كيف تؤثر القيمة على زيادة أو تقليل الحوادث.
    - استراتيجية: بناءً على هذه الميزات المتصدرة، يمكن لفرق الإنقاذ تركيز الموارد والطائرات بدون طيار 
      في الأيام التي تتجاوز فيها هذه المتغيرات الحد الحرج.
    """

    # 3. استخراج مخطط القوة (Force Plot) لحالة واحدة معينة
    # نختار الحالة ذات أعلى توقع للحوادث في العينة كحالة ذروة
    predictions = model.predict(X_sample)
    peak_idx = predictions.argmax()
    
    # Force Plot للحالة الأسوأ
    shap.initjs() # مطلوب إذا تم العرض في Jupyter، لكن هنا سنحفظه كـ HTML
    force_plot = shap.force_plot(explainer.expected_value, shap_values[peak_idx,:], X_sample.iloc[peak_idx,:])
    shap.save_html('shap_force_plot.html', force_plot)
    print("تم حفظ مخطط القوة في shap_force_plot.html")
    
    """
    تعليق لمتخذ القرار حول Force Plot:
    - يشرح هذا المخطط سبب وصول النموذج لهذا التنبؤ المحدد (في حالة ذروة معينة).
    - الأسهم باللون الأحمر (Pushing higher) تدفع عدد التنبؤات بالحوادث للارتفاع.
    - الأسهم باللون الأزرق (Pushing lower) تسحب التوقع للأسفل (عوامل أمان).
    - استراتيجية: في حالة محددة، إذا كان "ارتفاع الموج" هو المسبب الأكبر، يمكن توجيه
      أنظمة إنترنت الأشياء البحرية للتركيز على استشعار الأمواج وتنبيه الموانئ في الوقت الفعلي.
    """

if __name__ == "__main__":
    perform_shap_analysis()
