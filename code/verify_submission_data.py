"""Verify supplied analytical tables; this does not refit the original models."""
from pathlib import Path
import argparse,json,csv,math
from openpyxl import load_workbook

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('verification_results.json'));a=p.parse_args()
 b={i:load_workbook(a.data_dir/f'Supplementary_Data_{i}.xlsx',read_only=True,data_only=True) for i in range(1,5)}
 checks=[]
 def test(name,actual,expected,tol=0):
  passed=abs(actual-expected)<=tol;checks.append({'check':name,'actual':actual,'expected':expected,'tolerance':tol,'passed':passed})
 def n(i,s):return b[i][s].max_row-1
 for name,x,y in [('Biological sites',n(1,'Site_inputs'),130),('Biological site-events',n(1,'Sample_inputs'),564),('Biological responses',n(1,'Response_catalogue'),313),('Fish ecoregions',n(2,'Fish_units'),419),('Insect basins',n(2,'EPTO_units'),507),('Lake chlorophyll units',n(2,'Algae_units'),6003),('Water-quality station-years',sum(n(3,s) for s in ['DOSAT_station_year','EC_station_year','TP_station_year','NH4N_station_year','NO3N_station_year']),376295),('Water-quality range associations',n(4,'Fig3_water_quality'),78),('Biological response-domain associations',n(4,'Fig4_family_responses'),390),('All biological associations',n(4,'All_biological_associations'),16575),('Carrier associations',n(4,'All_carrier_associations'),12558),('Selected carrier rows',n(4,'Fig6_top_characteristics'),30)]:test(name,x,y)
 sheet=b[4]['Fig3_water_quality'];it=sheet.iter_rows(values_only=True);heads=next(it);records=[dict(zip(heads,row)) for row in it]
 for typ,extent,rho,q in [('River',4000,-0.646,0.0225),('Lake',500,-0.504,0.0364)]:
  row=next(x for x in records if x['water_body_type']==typ and x['environment_module']=='terrain' and x['range_m']==extent)
  test(typ+' topographic peak rho',row['spearman_rho'],rho,0.0005);test(typ+' topographic peak q',row['fdr_q'],q,0.00005)
 for wb in b.values():wb.close()
 a.output.write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'output':str(a.output)}))
 if not all(c['passed'] for c in checks):raise SystemExit(1)
if __name__=='__main__':main()
