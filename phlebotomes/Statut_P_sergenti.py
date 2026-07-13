
import pandas as pd
df = pd.read_csv(r'E:\leishpasteur\phlebotomes\phlebotomus_sergenti_par_province.csv')


count = df['Statut_P_sergenti'].value_counts().reset_index()
count.columns = ['Statut_P_sergenti', 'Nombre']
count['Pourcentage (%)'] = (count['Nombre'] / count['Nombre'].sum() * 100).round(2)

print("📊 RÉSUMÉ :\n")
print(count.to_string(index=False))

# Liste des provinces pour chaque statut
print("\n" + "="*80)
print("DÉTAIL PAR STATUT :\n")

for statut in df['Statut_P_sergenti'].unique():
    provinces = df[df['Statut_P_sergenti'] == statut]['Province'].unique()
    print(f"**{statut}** ({len(provinces)} provinces) :")
    print(", ".join(sorted(provinces)))
    print("-" * 50)