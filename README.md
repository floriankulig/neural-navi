
# neural-navi

Neural Navi is a IDSS (Intelligent Decision Support System) that analyses driving and camera data to provide insights and recommendations to drivers. The system is designed to help drivers make better decisions on the road, and to help them improve their driving skills.

## Setup

1. Create virtual environment:

   ```bash
   python -m venv venv --system-site-packages # Raspberry Pi OS (keeps preinstalled Picamera2-Package)
   # or
   python -m venv venv   # Windows/Mac
   ```

2. Activate virtual environment:

   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## What is this about?

### DE

Ziel dieses Projekts ist es mithilfe von Python einen CoPilot für das Fahren auf Autobahnen zu entwickeln. Dieser wird dabei iterativ mit Features ausgestattet, die das Fahren sicherer und effizienter machen sollen.
In der ersten Iteration soll der CoPilot dem Fahrer Empfehlungen zum Bremsen geben und damit gefährlichen Situationen aus dem Weg gehen und die Spritverbrauch zu optimieren.
Später sollen weitere Features hinzugefügt werden, wie z.B. Empfehlungen für Fahrspur- oder Gangwechsel.
Um das Umzusetzen, wird ein neuronales Netzwerk trainiert, das die Daten einer im Auto montierten Kamera und den Diagnostikdaten des Fahrzeugs analysiert und darauf basierend Empfehlungen gibt.

### EN

The goal of this project is to develop a CoPilot for highway driving using Python. It will iteratively add features to make driving safer and more efficient.
In the first iteration, the CoPilot will give the driver braking recommendations to avoid dangerous situations and optimize fuel consumption.
Other features will be added later, such as recommendations for lane and gear changes.
To do this, we are training a neural network that analyzes data from a camera mounted in the car and the car's diagnostic data, and makes recommendations based on this information.
