Notes on Using TestRunner with the Android Emulator
===================================================

 * Start the emulator from Android Studio
 * Forward the emulator's port 8080 to be available on the host machine as port 8081 with adb -s emulator-xxxx forward tcp:8081 tcp:8080
 * Then you can point the web browser on the local machine to localhost:8081 and see the OSD app web page (which is normally port 8080 on the device).

Therefore to set up three emulators to use different ports (8081-8083), do

adb -s emulator-5554 forward tcp:8081 tcp:8080
adb -s emulator-5556 forward tcp:8082 tcp:8080
adb -s emulator-5558 forward tcp:8083 tcp:8080



