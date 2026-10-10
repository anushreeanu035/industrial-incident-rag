# INCIDENT INVESTIGATION REPORT: INC-2023-044

**Machine ID:** M-301  
**Incident Type:** Electrical Motor Thermal Overload  
**Facility:** Bulk Material Handling Terminal, Yard Conveyor 3  
**Incident Date:** 2023-11-05  
**Severity Level:** Tier 2 Operational Interruption  
**Lead Investigator:** Electrical Maintenance Supervisor James K. O'Connor  

---

## 1. Overview of Incident
At 09:18 on November 5, 2023, the 400 kW 3.3 kV induction drive motor M-301 powering Yard Stacker Conveyor 3 tripped unexpectedly on protection code E-301 (Motor Thermal Overload). Stator winding Embedded Temperature Detectors (ETD-1 through ETD-6) registered a maximum temperature of 148°C, approaching the Class F winding insulation thermal limit of 155°C. The sudden stoppage stranded 650 tons of iron ore pellet on the belt.

## 2. Sequence of Observations
- **08:30:** Operating current on phase B was 84 A (normal nominal current 78 A).
- **09:00:** Ambient yard temperature reached 38°C with high relative humidity.
- **09:12:** High temperature warning pre-alarm enunciated at 135°C on motor drive terminal.
- **09:18:** Thermal overload relay tripped motor breaker; conveyor stopped.

## 3. Physical Inspection & Diagnostics
1. **Airflow Restriction:** Motor stator cooling fins were covered in a compacted crust of iron ore dust and grease slurry measuring 15 mm to 25 mm in thickness across 75% of the external frame circumference. Cooling fan cowl air intake screen was 60% blocked with dried pulp residue.
2. **Bearing Degradation:** Ultrasonic acoustic emissions analysis on drive-end roller bearing B-301 indicated a crest factor of 6.2 with peak amplitude of 52 dBµV, characteristic of boundary lubrication regime. Grease sample extraction showed severe oxidation, hardening, and high particulate content.
3. **Electrical Insulation Testing:** Megohmmeter insulation resistance test of stator windings measured 450 Megaohms (acceptable threshold > 100 Megaohms), verifying that no dielectric breakdown of the winding enamel had occurred.

## 4. Root Cause Determination
- **Primary Root Cause:** Severe external thermal choking caused by heavy dust accumulation on stator cooling ribs. The thermal resistance of the dust layer reduced convective air cooling efficiency by an estimated 70%, preventing thermal dissipation under sustained full-load duty.
- **Secondary Root Cause:** Lubricant dry-out in bearing B-301 generated parasitic frictional heating that transferred conducted heat directly into the rotor shaft and front end-shield.

## 5. Corrective Actions Implemented
1. Performed thorough dry-ice blast cleaning of motor cooling ribs, fan cowl, and intake cowlings.
2. Replaced drive-end bearing B-301 and flushed housing with high-temperature polyurea synthetic grease (Kluberplex BEM 41-132).
3. Revised mechanical PM schedule PM-E-301 to mandate bi-weekly pressurized air washdown of exterior motor fins.
4. Installed wireless continuous tri-axial vibration and surface temperature telemetry pod (sensor VM-301) with live threshold alarms configured at 120°C.
