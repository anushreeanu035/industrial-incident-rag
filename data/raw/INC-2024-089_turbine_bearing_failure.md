# INCIDENT INVESTIGATION REPORT: INC-2024-089

**Machine ID:** HP-17  
**Incident Type:** Mechanical Vibration & Bearing Failure  
**Facility:** Generation Unit 2, Thermal Power Station  
**Incident Date:** 2024-03-14  
**Severity Level:** Tier 1 Critical Shutdown  
**Report Author:** Senior Reliability Engineer Marcus Vance  

---

## 1. Executive Summary
On March 14, 2024, at 03:42 UTC, high-pressure steam turbine HP-17 tripped automatically on high radial shaft vibration. Radial shaft vibration on non-drive end (NDE) journal bearing B-101 spiked rapidly from a nominal 2.1 mm/s RMS to 14.2 mm/s RMS within 90 seconds, exceeding the emergency trip threshold of 9.0 mm/s. No personnel injuries occurred, but Generation Unit 2 was taken offline for 72 hours.

## 2. Chronological Sequence of Events
- **03:20 UTC:** Lube oil header temperature increased from 46°C to 58°C.
- **03:31 UTC:** Minor vibration increase detected on bearing B-101 (3.4 mm/s RMS).
- **03:40 UTC:** Bearing metal temperature transmitter TT-101 registered 114°C (alarm setpoint 105°C).
- **03:41 UTC:** Lube oil supply pressure downstream of orifice valve V-204 dropped from 2.4 bar to 0.7 bar.
- **03:42 UTC:** Vibration spiked to 14.2 mm/s RMS; automated protection relay 88V executed turbine emergency trip.

## 3. Physical Evidence and Inspection Findings
Disassembly of bearing B-101 revealed:
1. Severe wiping and smearing of the lead-tin-copper babbitt lining on lower tilting pads 3 and 4.
2. Babbitt metal flakes accumulated in the oil return pocket.
3. Laboratory oil analysis of sample LO-89 indicated 850 ppm moisture contamination and ethylene glycol traces.
4. Orifice control valve V-204 internal metering port was 80% occluded by carbonized sludge and particulate debris.
5. Shell-and-tube lube oil heat exchanger E-201 displayed a degraded internal elastomeric O-ring gasket, allowing coolant water to bypass into the oil channel.

## 4. Root Cause Analysis
- **Primary Root Cause:** Seal degradation in oil cooler E-201 allowed cooling water and glycol to contaminate the turbine lubrication circuit. The resulting emulsion accelerated sludge accumulation, leading to particulate blockage of orifice valve V-204. Starved of pressurized oil film, hydrodynamic lubrication collapsed, causing babbitt wiping and severe shaft rub.
- **Contributing Factors:** Inadequate differential pressure telemetry across pre-filter F-101, which failed to warn operators of filter bypass during high flow demand.

## 5. Corrective and Preventive Actions
1. Replaced journal bearing set B-101 with new precision-machined tilting pad assemblies.
2. Performed full chemical flush and dehydration of 4,000-liter turbine oil header system.
3. Replaced O-ring gaskets on heat exchanger E-201 with high-temperature fluoroelastomer seals.
4. Installed differential pressure transmitter PDT-101 across duplex pre-filter F-101 with automated alarm at 1.2 bar delta-P.
5. Instituted mandatory 500-hour particulate fluid sampling and quarterly cooler eddy current testing.
