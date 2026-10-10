# INCIDENT INVESTIGATION REPORT: INC-2024-201

**Machine ID:** K-102  
**Incident Type:** Aerodynamic Compressor Surge & Tripping  
**Facility:** Olefins Cracking Unit, Train 1  
**Incident Date:** 2024-05-18  
**Severity Level:** Tier 1 Process Interruption  
**Lead Investigator:** Turbomachinery Specialist Arthur Pendelton  

---

## 1. Executive Summary
On May 18, 2024, at 22:04 local time, the multi-stage centrifugal cracked gas compressor K-102 suffered repeated aerodynamic surge cycles, triggering automated emergency shutdown ESD-1 via the anti-surge controller. Dynamic differential pressure sensors recorded surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz, placing severe axial thrust loads on active thrust bearing assembly TB-102.

## 2. Event Sequence
- **21:55:** Ambient temperature dropped sharply following a heavy convective rainstorm.
- **22:01:** Instrument air header pressure fluctuation noticed in Train 1 compressor house.
- **22:03:** Suction header bypass valve FV-102 slammed shut without modulation command from the DCS.
- **22:03:45:** Suction flow fell precipitously below the compressor minimum aerodynamic stability limit (surge line).
- **22:04:10:** Compressor entered deep surge. Anti-surge recycle valve ASV-101 failed to open fast enough to arrest flow collapse.
- **22:04:18:** Automated vibration and reverse-flow protection triggered trip ESD-1.

## 3. Physical Inspection & Diagnostics
1. **Instrument Air System Moisture:** Examination of instrument air supply header revealed excessive liquid water accumulation. The upstream twin-tower heatless desiccant air dryer D-10 had experienced a switching valve failure (valve SV-4 stuck mid-stroke), allowing saturated unconditioned air into the distribution lines.
2. **Pilot Solenoid Freeze-up:** Pilot solenoid valve SOV-102 controlling suction bypass valve FV-102 was found frozen in the de-energized exhaust position due to water droplet crystallization in the exhaust port orifice.
3. **Thrust Bearing Inspection:** Rotor axial displacement sensor registered 0.42 mm deflection (normal < 0.15 mm). Inspection of active thrust shoes on TB-102 confirmed babbitt fatigue micro-cracking and minor scuffing due to repetitive transient surge loading.

## 4. Root Causes
- **Direct Cause:** Spurious closing of bypass valve FV-102 starved compressor K-102 of adequate mass flow, driving the operating point over the surge margin.
- **Root Cause:** Failure of desiccant air dryer D-10 switching valve SV-4 allowed high dew-point air into outdoor instrument tubing, causing icing and pneumatic solenoid failure during the ambient temperature drop.
- **Systemic Factor:** Anti-surge valve ASV-101 actuator had an overly conservative booster volume setting, resulting in a 1.4-second slew time when the design specification called for under 750 milliseconds.

## 5. Corrective Actions Taken
1. Replaced switching valve SV-4 and recharged desiccant alumina bed on air dryer D-10.
2. Installed self-regulating electrical heat tracing and closed-cell elastomeric insulation on all instrument air lines feeding critical turbomachinery valves.
3. Replaced thrust bearing shoes on TB-102 with copper-backed babbitt pads.
4. Upgraded pneumatic booster on anti-surge valve ASV-101, reducing full-stroke opening time from 1.4 seconds to 0.58 seconds.
