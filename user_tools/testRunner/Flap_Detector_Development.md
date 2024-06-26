# Flap Detector

## Background
Recently Benjamin has been having more seizures, but rather than being full tonic-clonic ones, they have been partial ones where he has remained kneeling up and his left arm has 'flapped' (maybe 2Hz?) rather than rapid shaking that we have the default OSD algorithm tuned to.

I have been using a more sensitive set up for the OSD algorithm with the frequency range set to 2-8Hz rather than the default 3-8 Hz, and have also set down the Alarm Ratio threshold to 54 rather than the default 57.   This has given us a decent 84% detection reliability for all of Benjamin's seizures, compared to 67% with the default settings.

The problem is it has also given us a much higher false alarm rate - benjamin walking slowly can now set it off, as does scratching his head in the night....so I want to reduce the false alarm rate......

## Methodology
Benjamin is using the PineTime watch seizure detector, so to make sure I had a good set of 'normal' activities measured using his watch, we logged normal daily activities (NDA) events for about 24 hours (excluding the time he is out of the house) - this gave us 326 NDA events to use to assess false alarm performance.   We also used all of his seizure data from the Open Seizure Database

Testrunner was set up to analyse the default OSD settings, the current settings we are using, plus a 'flap' detector set-up where we set the range to 2-3Hz and set the Alarm Ratio threshold to a high value (1000) so we are tuned very much to that flapping motion.

We then compare the seizure detection reliability and the number of false alarms from the NDA activities for the three algorithms.  For the purposes of this analysis we define successful detection as a full ALARM initiation, and a false alarm as a full ALARM initiation (ie we ignore events that only generate WARNING alerts).

# Results

## Baseline

  - Default OSD settings (3-8 Hz, Alarm Ratio Threshold=57, Alarm Threshold=100) gave a 67% detection reliability, and 98% of NDA events correct (7 out of 326 events gave false alarms).
  - Current Settings (2-8 Hz, Alarm Ratio Threshold=54, Alarm Threshold=500) gae 84% detection reliability, and 97% of NDA events (9 out of 326 events gave false alarms).

## Flap Detection

The flap detection settings were run with several variations on the precise settings, varying both the frequency range and the alarm thresholds as shown in the table below.   This was done by manual optimisation using judgement - we can probably do better by automating it and varying both the OSD and the flap algorithm settings.

From the manual optimisation shown below it appears that using the default OSD settings in addition to a 'flap' detector that uses a frequency range of 2-4 Hz, an alarm threshold of 5000 and a ratio threshold of 90 will give us an improved detection reliability for Benjamin's seizures, with only a small increase in false alarms, which is much better than the settings we are currently using.

| Version | Freq Range (Hz) | Alarm Ratio Threshold | Alarm Threshold | Overall Detection Reliability | 'Flap' Detection Reliability | NDA Correct |
| 1       | 2 - 3     | 80 | 100 | 67%  | --- | 72% | It was detecting seizures, but a lot of false alarms
| 2       | 2 - 3     | 80 | 1000 | 55%  | --- | 78% | Seizure detection looks worse, but this is ok if we are detecting the 'flap' seizures, and NDA performance is getting better.  Increaes threshold further so we only detect large movements.
| 3       | 2 - 3     | 80 | 10000 | 30%  | 50% | 99% | So we are getting a reasonable NDA performance now.   This detects 8 'flap' seizures which were not detected by default settings, which would increase the overall detection reliability using the default settings to 80%, but gave two additional false alarms.   Try reducing the alarm threshold and increasing the alarm ratio threshold.
| 4       | 2 - 3     | 90 | 5000 | 39%  | --- | 96% | false alarms deteriorated compared to V3, so not analysed
| 5       | 2 - 4     | 80 | 10000 | 31% | 54% | 99% | Detected 9 additional seizures that default settings missed, which gets to 81% combined reliability.   But still only detects 54% of seizures labelled 'flap' - try lowering frequency range.
| 6       | 1 - 3     | 80 | 10000 | 28% | 50% | 97% | Now detects 12 additional seizures (86%), but more false alarms.
| 7       | 1 - 3     | 90 | 10000 | 25% | % | 97% | Incrreased threshold to try to reduce false alarms, but it just reduced detection reliability.
| 8       | 1 - 4     | 80 | 10000 | 30% | 50% | 98% | As for Version 5, but lower frequency reduced.  Version 5 is still the best so far.
| 9       | 2 - 4     | 80 | 5000 | 41% | 54% | 97% | As for Version 5, but lower alarm threshold.  Detection reliability better, but also results in 10 additional false alarms, so not suitable.   Version 5 still looking best.   Try increasing ratio threshold to see if that drops false alarms before giving up on manual optimisation.
| 10       | 2 - 4     | 90 | 5000 | 33% | 42% | 99% | As for Version 5, but lower alarm threshold and higher ratio threshold.  Detected 10 additional seizures.   Although flap detection reliability is low for just the flap algorithm, the default osd plus flap algorithm give 85% detection reliability.   It does give 3 additional NDA alarms compared to default settings, but these were not middle of the night ones, so less of a problem - our current settings gave 8 additional alarms, including 2 in the middle of the night.

#Conclusion

To improve the detetion reliability of 'flapping' type seizures without resulting on a lot of additional false alarms, we want to run 2 versions of the OSD algorithm.   One will use the default settings, and the other will be tuned to lower frequency, but have a higher threshold to reduce false alarms.