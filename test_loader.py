# test_loader.py

from src.data_loader import FOSTER2025Loader
import matplotlib.pyplot as plt

def test_foster_loader():
    """
    اختبار تحميل FOSTER dataset
    """
    print("="*60)
    print("Testing FOSTER 2025 Data Loader")
    print("="*60)
    
    # إنشاء loader
    loader = FOSTER2025Loader()
    
    # تحميل 5 تسجيلات للاختبار
    data = loader.load_all(max_samples=5)
    
    if not data:
        print("No data loaded! Check dataset path.")
        return
    
    # عرض أول تسجيل
    print("\n" + "="*60)
    print("First Recording Details:")
    print("="*60)
    
    sample = data[0]
    for key, value in sample.items():
        if key != 'signal':  # لا نطبع الإشارة (كبيرة جداً)
            print(f"{key}: {value}")
    
    # رسم أول 3 تسجيلات
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    for i in range(min(3, len(data))):
        ax = axes[i]
        signal = data[i]['signal']
        
        # ارسم أول 5 ثواني فقط
        samples_to_plot = min(len(signal), 5 * data[i]['fs'])
        
        ax.plot(signal[:samples_to_plot])
        ax.set_title(f"Recording {i+1}: {data[i]['filename']}")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/results/foster_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n Plots saved to data/results/foster_samples.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    test_foster_loader()