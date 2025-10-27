# PCB Overview

The Aero Hand consists of two custom PCB boards: **Board A** and **Board B**. All design files - including Gerber files, KiCad project files, BOM, and CPL - are available in our [GitHub Repository Link](<https://github.com/TetherIA/aero-hand-open/tree/main/hardware>).

---

## Board A
- Features **8 Molex connectors** for connecting all servos in the hand.
- Includes a **JST connector P3.96mm** for outer wiring and fitting inside the hand enclosure.
- Designed for compact integration and reliable servo connectivity.
![Board A schematic showing 8 Molex connectors and JST connector P3.96mm](./imgs/boardaschematic.png)
![Board A PCB layout view](./imgs/boardalayout.png)
![Board A 3D rendered view](./imgs/boarda3d.png)

## Board B
- Includes a **JST connector** and **one Molex connector** for testing individual servos and debugging.
- Features a **terminal block** to connect a regulated 6V, 10A power supply for the servos.
- Hosts the **ESP32-S3** microcontroller with a **USB-C port** for:
	- Powering the ESP32-S3
	- Communication between PC and ESP32-S3
	- Interfacing with the hand
- Board B is essential for diagnostics and safe power delivery.
![PCB Schematic Board B](./imgs/boardbschematic.png)
![PCB layout Board B](./imgs/boardblayout.png)
![Board B 3D rendered view](./imgs/boardb3d.png)

---

For more details on communication and control, refer to our [SDK Documentation](https://github.com/TetherIA/aero-open-sdk).

<div align="center">

Made with ❤️ by **TetherIA Robotics**

</div>