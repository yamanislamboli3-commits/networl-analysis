from scapy.all import Ether, IP, TCP, wrpcap
import time

def generate_simulation_pcap(filename="simulasyon_trafigi.pcap"):
    packets = []
    
    # Zaman damgalarını (timestamp) gerçekçi tutmak için şu anki zamanı baz alıyoruz
    base_time = time.time()

   

    print("2. Zararlı trafik (SYN Flood DoS) oluşturuluyor...")
    # Saldırı Akışı: Farklı bir IP'den, aynı hedefe 1 milisaniye arayla gönderilen 2000 adet SYN paketi
    attack_start_time = base_time  # Normal trafikten 15 saniye sonra başlasın
    
    for i in range(300):
        # sport (Kaynak portu) sürekli değişiyor, flags="S" (Sadece SYN bayrağı)
        pkt = Ether() / IP(src="10.10.10.50", dst="192.168.1.100") / TCP(sport=1024 + (i % 60000), dport=80, flags="S")
        pkt.time = attack_start_time + (i * 0.001) # Her paket arası 1 milisaniye (Çok hızlı)
        packets.append(pkt)

    print(f"\nToplam {len(packets)} paket bellekte oluşturuldu.")
    print(f"'{filename}' dosyasına yazılıyor...")
    
    # Ağa yollama (send) işlemi YAPILMAZ. Doğrudan diske dosya olarak yazılır.
    wrpcap(filename, packets)
    print("✅ İşlem başarıyla tamamlandı! Dosyayı CICFlowMeter veya Wireshark ile inceleyebilirsiniz.")

if __name__ == "__main__":
    generate_simulation_pcap()