from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import os
import posixpath
import re
import zipfile
from copy import deepcopy
from xml.etree import ElementTree as ET


# =========================
# 1. Criar resultado da partida
# =========================

def definir_resultado(row):
    if row["team_a_win"] == 1:
        return 1   # Time A venceu
    elif row["team_b_win"] == 1:
        return -1  # Time B venceu
    else:
        return 0   # Empate

df = pd.read_csv("matches.csv")


df["resultado"] = df.apply(definir_resultado, axis=1)


# =========================
# 2. Criar estatísticas dos times
# =========================

time_a = df[[
    "team_a_name",
    "team_a_score",
    "team_b_score",
    "team_a_win",
    "team_b_win",
    "draw"
]].copy()

time_a.columns = [
    "team",
    "gols_marcados",
    "gols_sofridos",
    "vitoria",
    "derrota",
    "empate"
]

time_b = df[[
    "team_b_name",
    "team_b_score",
    "team_a_score",
    "team_b_win",
    "team_a_win",
    "draw"
]].copy()

time_b.columns = [
    "team",
    "gols_marcados",
    "gols_sofridos",
    "vitoria",
    "derrota",
    "empate"
]

historico_times = pd.concat([time_a, time_b])

stats = historico_times.groupby("team").agg(
    jogos=("team", "count"),
    media_gols_marcados=("gols_marcados", "mean"),
    media_gols_sofridos=("gols_sofridos", "mean"),
    taxa_vitoria=("vitoria", "mean"),
    taxa_empate=("empate", "mean"),
    taxa_derrota=("derrota", "mean")
).reset_index()


# =========================
# 3. Montar base para o modelo
# =========================

df_modelo = df.merge(
    stats,
    left_on="team_a_name",
    right_on="team",
    how="left"
)

df_modelo = df_modelo.rename(columns={
    "jogos": "a_jogos",
    "media_gols_marcados": "a_media_gols_marcados",
    "media_gols_sofridos": "a_media_gols_sofridos",
    "taxa_vitoria": "a_taxa_vitoria",
    "taxa_empate": "a_taxa_empate",
    "taxa_derrota": "a_taxa_derrota"
}).drop(columns=["team"])

df_modelo = df_modelo.merge(
    stats,
    left_on="team_b_name",
    right_on="team",
    how="left"
)

df_modelo = df_modelo.rename(columns={
    "jogos": "b_jogos",
    "media_gols_marcados": "b_media_gols_marcados",
    "media_gols_sofridos": "b_media_gols_sofridos",
    "taxa_vitoria": "b_taxa_vitoria",
    "taxa_empate": "b_taxa_empate",
    "taxa_derrota": "b_taxa_derrota"
}).drop(columns=["team"])

df_modelo["diff_gols_marcados"] = (
    df_modelo["a_media_gols_marcados"] - df_modelo["b_media_gols_marcados"]
)

df_modelo["diff_gols_sofridos"] = (
    df_modelo["a_media_gols_sofridos"] - df_modelo["b_media_gols_sofridos"]
)

df_modelo["diff_taxa_vitoria"] = (
    df_modelo["a_taxa_vitoria"] - df_modelo["b_taxa_vitoria"]
)


# =========================
# 4. Treinar modelo
# =========================

features = [
    "a_jogos",
    "a_media_gols_marcados",
    "a_media_gols_sofridos",
    "a_taxa_vitoria",
    "a_taxa_empate",
    "a_taxa_derrota",

    "b_jogos",
    "b_media_gols_marcados",
    "b_media_gols_sofridos",
    "b_taxa_vitoria",
    "b_taxa_empate",
    "b_taxa_derrota",

    "diff_gols_marcados",
    "diff_gols_sofridos",
    "diff_taxa_vitoria"
]

X = df_modelo[features].fillna(0)
y = df_modelo["resultado"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

print("\nRelatório de avaliação do modelo:")
print(classification_report(y_test, pred))


# =========================
# 5. Função para prever uma partida
# =========================

def prever_partida(time_a_nome, time_b_nome):
    if time_a_nome not in stats["team"].values:
        raise ValueError(f"Time não encontrado na base: {time_a_nome}")

    if time_b_nome not in stats["team"].values:
        raise ValueError(f"Time não encontrado na base: {time_b_nome}")

    a = stats[stats["team"] == time_a_nome].iloc[0]
    b = stats[stats["team"] == time_b_nome].iloc[0]

    entrada = pd.DataFrame([{
        "a_jogos": a["jogos"],
        "a_media_gols_marcados": a["media_gols_marcados"],
        "a_media_gols_sofridos": a["media_gols_sofridos"],
        "a_taxa_vitoria": a["taxa_vitoria"],
        "a_taxa_empate": a["taxa_empate"],
        "a_taxa_derrota": a["taxa_derrota"],

        "b_jogos": b["jogos"],
        "b_media_gols_marcados": b["media_gols_marcados"],
        "b_media_gols_sofridos": b["media_gols_sofridos"],
        "b_taxa_vitoria": b["taxa_vitoria"],
        "b_taxa_empate": b["taxa_empate"],
        "b_taxa_derrota": b["taxa_derrota"],

        "diff_gols_marcados": a["media_gols_marcados"] - b["media_gols_marcados"],
        "diff_gols_sofridos": a["media_gols_sofridos"] - b["media_gols_sofridos"],
        "diff_taxa_vitoria": a["taxa_vitoria"] - b["taxa_vitoria"]
    }])

    predicao = modelo.predict(entrada)[0]
    probabilidades = modelo.predict_proba(entrada)[0]
    classes = modelo.classes_

    resultado_prob = dict(zip(classes, probabilidades))

    return predicao, resultado_prob


ALIASES_TIMES = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Rep. of Korea": "South Korea",
    "Czech Rep.": "Czech Republic",
    "Bosnia/Herzeg.": "Bosnia and Herzegovina",
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "Zaire",
}


def normalizar_nome_time(nome):
    return ALIASES_TIMES.get(nome, nome)


def obter_stats_time(nome):
    nome_normalizado = normalizar_nome_time(nome)

    if nome_normalizado in stats["team"].values:
        return stats[stats["team"] == nome_normalizado].iloc[0]

    medias = stats.mean(numeric_only=True)
    return pd.Series({
        "team": nome_normalizado,
        "jogos": 0,
        "media_gols_marcados": medias["media_gols_marcados"],
        "media_gols_sofridos": medias["media_gols_sofridos"],
        "taxa_vitoria": medias["taxa_vitoria"],
        "taxa_empate": medias["taxa_empate"],
        "taxa_derrota": medias["taxa_derrota"],
    })


def prever_placar(time_a_nome, time_b_nome):
    a = obter_stats_time(time_a_nome)
    b = obter_stats_time(time_b_nome)

    entrada = pd.DataFrame([{
        "a_jogos": a["jogos"],
        "a_media_gols_marcados": a["media_gols_marcados"],
        "a_media_gols_sofridos": a["media_gols_sofridos"],
        "a_taxa_vitoria": a["taxa_vitoria"],
        "a_taxa_empate": a["taxa_empate"],
        "a_taxa_derrota": a["taxa_derrota"],

        "b_jogos": b["jogos"],
        "b_media_gols_marcados": b["media_gols_marcados"],
        "b_media_gols_sofridos": b["media_gols_sofridos"],
        "b_taxa_vitoria": b["taxa_vitoria"],
        "b_taxa_empate": b["taxa_empate"],
        "b_taxa_derrota": b["taxa_derrota"],

        "diff_gols_marcados": a["media_gols_marcados"] - b["media_gols_marcados"],
        "diff_gols_sofridos": a["media_gols_sofridos"] - b["media_gols_sofridos"],
        "diff_taxa_vitoria": a["taxa_vitoria"] - b["taxa_vitoria"]
    }])

    resultado = modelo.predict(entrada)[0]

    gols_a = round((a["media_gols_marcados"] + b["media_gols_sofridos"]) / 2)
    gols_b = round((b["media_gols_marcados"] + a["media_gols_sofridos"]) / 2)

    gols_a = max(0, min(5, int(gols_a)))
    gols_b = max(0, min(5, int(gols_b)))

    if resultado == 1 and gols_a <= gols_b:
        gols_a = gols_b + 1
    elif resultado == -1 and gols_b <= gols_a:
        gols_b = gols_a + 1
    elif resultado == 0:
        empate = max(0, min(3, round((gols_a + gols_b) / 2)))
        gols_a = empate
        gols_b = empate

    return gols_a, gols_b


def criar_planilha_previsoes(previsoes, arquivo_saida="previsoes_copa_2026.xlsx"):
    def coluna_para_letra(numero):
        texto = ""
        while numero:
            numero, resto = divmod(numero - 1, 26)
            texto = chr(65 + resto) + texto
        return texto

    def escape_xml(valor):
        texto = "" if valor is None else str(valor)
        return (
            texto.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    cabecalhos = [
        "Jogo",
        "Grupo",
        "Time A",
        "Gols A",
        "Gols B",
        "Time B",
        "Resultado previsto",
    ]

    linhas = [cabecalhos]
    for previsao in previsoes:
        linhas.append([
            previsao["jogo"],
            previsao["grupo"],
            previsao["time_a"],
            previsao["gols_a"],
            previsao["gols_b"],
            previsao["time_b"],
            previsao["resultado"],
        ])

    linhas_xml = []
    for indice_linha, linha in enumerate(linhas, start=1):
        celulas_xml = []
        for indice_coluna, valor in enumerate(linha, start=1):
            referencia = f"{coluna_para_letra(indice_coluna)}{indice_linha}"
            estilo = ' s="1"' if indice_linha == 1 else ""
            if isinstance(valor, int):
                celulas_xml.append(f'<c r="{referencia}"{estilo}><v>{valor}</v></c>')
            else:
                celulas_xml.append(
                    f'<c r="{referencia}" t="inlineStr"{estilo}>'
                    f"<is><t>{escape_xml(valor)}</t></is></c>"
                )
        linhas_xml.append(f'<row r="{indice_linha}">{"".join(celulas_xml)}</row>')

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:G{len(linhas)}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <cols>
    <col min="1" max="1" width="8" customWidth="1"/>
    <col min="2" max="2" width="10" customWidth="1"/>
    <col min="3" max="3" width="24" customWidth="1"/>
    <col min="4" max="5" width="10" customWidth="1"/>
    <col min="6" max="6" width="24" customWidth="1"/>
    <col min="7" max="7" width="30" customWidth="1"/>
  </cols>
  <sheetData>
    {"".join(linhas_xml)}
  </sheetData>
  <autoFilter ref="A1:G{len(linhas)}"/>
</worksheet>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Previsoes_IA" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    with zipfile.ZipFile(arquivo_saida, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", rels)
        xlsx.writestr("xl/workbook.xml", workbook_xml)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/styles.xml", styles_xml)
        xlsx.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    return arquivo_saida


def preencher_planilha_copa(
    arquivo_entrada="copa 2026WCup_2026_4.2.5_en.xlsx",
    arquivo_saida="copa_2026_preenchida.xlsx"
):
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def coluna_para_numero(coluna):
        total = 0
        for letra in coluna:
            total = total * 26 + ord(letra) - ord("A") + 1
        return total

    def numero_para_coluna(numero):
        texto = ""
        while numero:
            numero, resto = divmod(numero - 1, 26)
            texto = chr(65 + resto) + texto
        return texto

    def dividir_celula(referencia):
        match = re.fullmatch(r"([A-Z]+)(\d+)", referencia)
        return match.group(1), int(match.group(2))

    def valor_celula(celula, shared_strings):
        tipo = celula.attrib.get("t")

        if tipo == "inlineStr":
            partes = celula.findall(".//main:t", ns)
            return "".join(parte.text or "" for parte in partes)

        v = celula.find("main:v", ns)
        if v is None or v.text is None:
            return None

        if tipo == "s":
            return shared_strings[int(v.text)]

        texto = v.text
        try:
            numero = float(texto)
            if numero.is_integer():
                return int(numero)
            return numero
        except ValueError:
            return texto

    def carregar_shared_strings(zip_arquivo):
        if "xl/sharedStrings.xml" not in zip_arquivo.namelist():
            return []

        raiz = ET.fromstring(zip_arquivo.read("xl/sharedStrings.xml"))
        strings = []
        for item in raiz.findall("main:si", ns):
            partes = item.findall(".//main:t", ns)
            strings.append("".join(parte.text or "" for parte in partes))
        return strings

    def caminhos_planilhas(zip_arquivo):
        workbook = ET.fromstring(zip_arquivo.read("xl/workbook.xml"))
        rels = ET.fromstring(zip_arquivo.read("xl/_rels/workbook.xml.rels"))

        id_para_caminho = {}
        for relacao in rels.findall("pkgrel:Relationship", ns):
            alvo = relacao.attrib["Target"]
            caminho = alvo if alvo.startswith("xl/") else posixpath.normpath("xl/" + alvo)
            id_para_caminho[relacao.attrib["Id"]] = caminho

        mapa = {}
        for planilha in workbook.findall("main:sheets/main:sheet", ns):
            nome = planilha.attrib["name"]
            rel_id = planilha.attrib[f"{{{ns['rel']}}}id"]
            mapa[nome] = id_para_caminho[rel_id]
        return mapa

    def ler_planilha(zip_arquivo, caminho, shared_strings):
        raiz = ET.fromstring(zip_arquivo.read(caminho))
        valores = {}
        for celula in raiz.findall(".//main:c", ns):
            ref = celula.attrib.get("r")
            if ref:
                valores[ref] = valor_celula(celula, shared_strings)
        return valores

    def setar_numero(raiz, referencia, valor):
        sheet_data = raiz.find("main:sheetData", ns)
        coluna, linha = dividir_celula(referencia)

        linhas = {
            int(row.attrib["r"]): row
            for row in sheet_data.findall("main:row", ns)
        }

        if linha in linhas:
            row = linhas[linha]
        else:
            row = ET.Element(f"{{{ns['main']}}}row", {"r": str(linha)})
            sheet_data.append(row)

        celulas = {cell.attrib["r"]: cell for cell in row.findall("main:c", ns)}
        if referencia in celulas:
            celula = celulas[referencia]
        else:
            celula = ET.Element(f"{{{ns['main']}}}c", {"r": referencia})
            row.append(celula)

        celula.attrib.pop("t", None)

        formula = celula.find("main:f", ns)
        if formula is not None:
            celula.remove(formula)

        v = celula.find("main:v", ns)
        if v is None:
            v = ET.SubElement(celula, f"{{{ns['main']}}}v")
        v.text = str(valor)

        row[:] = sorted(
            row.findall("main:c", ns),
            key=lambda cell: coluna_para_numero(dividir_celula(cell.attrib["r"])[0])
        )

    with zipfile.ZipFile(arquivo_entrada, "r") as origem:
        shared_strings = carregar_shared_strings(origem)
        mapas = caminhos_planilhas(origem)

        language = ler_planilha(origem, mapas["Language"], shared_strings)
        groups = ler_planilha(origem, mapas["Groups"], shared_strings)
        matches = ler_planilha(origem, mapas["Matches"], shared_strings)

        world_path = mapas["World Cup"]
        world = ler_planilha(origem, world_path, shared_strings)
        world_root = ET.fromstring(origem.read(world_path))

        numero_para_time = {}
        for linha in range(5, 101):
            numero = language.get(f"D{linha}")
            nome = language.get(f"F{linha}")
            if numero and nome:
                numero_para_time[numero] = nome

        posicao_para_time = {}
        for linha in range(7, 66):
            posicao = groups.get(f"B{linha}")
            numero = groups.get(f"C{linha}")
            if posicao and numero in numero_para_time:
                posicao_para_time[posicao] = numero_para_time[numero]

        celulas_placares = {}
        linhas_jogos = [11, 16, 21, 26, 31, 36]
        colunas_jogos = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]

        for linha in linhas_jogos:
            for coluna in colunas_jogos:
                referencia = f"{numero_para_coluna(coluna)}{linha}"
                numero_jogo = world.get(referencia)
                if isinstance(numero_jogo, int):
                    celulas_placares[numero_jogo] = (
                        f"{numero_para_coluna(coluna + 1)}{linha + 3}",
                        f"{numero_para_coluna(coluna + 2)}{linha + 3}"
                    )

        jogos_preenchidos = 0
        previsoes = []

        for linha in range(4, 76):
            numero_jogo = matches.get(f"B{linha}")
            posicao_a = matches.get(f"C{linha}")
            posicao_b = matches.get(f"D{linha}")

            if numero_jogo not in celulas_placares:
                continue

            time_a = posicao_para_time.get(posicao_a)
            time_b = posicao_para_time.get(posicao_b)

            if not time_a or not time_b:
                continue

            gols_a, gols_b = prever_placar(time_a, time_b)
            celula_a, celula_b = celulas_placares[numero_jogo]
            setar_numero(world_root, celula_a, gols_a)
            setar_numero(world_root, celula_b, gols_b)

            grupo = posicao_a[0] if isinstance(posicao_a, str) else ""
            if gols_a > gols_b:
                resultado = f"Vitoria de {time_a}"
            elif gols_b > gols_a:
                resultado = f"Vitoria de {time_b}"
            else:
                resultado = "Empate"

            previsoes.append({
                "jogo": numero_jogo,
                "grupo": grupo,
                "time_a": time_a,
                "gols_a": gols_a,
                "gols_b": gols_b,
                "time_b": time_b,
                "resultado": resultado,
            })
            jogos_preenchidos += 1

        ET.register_namespace("", ns["main"])
        ET.register_namespace("r", ns["rel"])
        world_xml = ET.tostring(world_root, encoding="utf-8", xml_declaration=True)

        if os.path.abspath(arquivo_entrada) == os.path.abspath(arquivo_saida):
            raise ValueError("Use um arquivo de saida diferente do arquivo original.")

        with zipfile.ZipFile(arquivo_saida, "w", compression=zipfile.ZIP_DEFLATED) as destino:
            for item in origem.infolist():
                conteudo = world_xml if item.filename == world_path else origem.read(item.filename)
                info = deepcopy(item)
                destino.writestr(info, conteudo)

    arquivo_previsoes = criar_planilha_previsoes(previsoes)
    return jogos_preenchidos, arquivo_saida, arquivo_previsoes


# =========================
# 6. Teste de previsão
# =========================

time_a_teste = "Brazil"
time_b_teste = "France"

resultado, probabilidades = prever_partida(time_a_teste, time_b_teste)

print(f"\nPrevisão para {time_a_teste} x {time_b_teste}:")

if resultado == 1:
    print(f"Resultado previsto: vitória de {time_a_teste}")
elif resultado == -1:
    print(f"Resultado previsto: vitória de {time_b_teste}")
else:
    print("Resultado previsto: empate")

print("\nProbabilidades:")
print(f"{time_a_teste} vencer: {probabilidades.get(1, 0):.2%}")
print(f"Empate: {probabilidades.get(0, 0):.2%}")
print(f"{time_b_teste} vencer: {probabilidades.get(-1, 0):.2%}")


# =========================
# 7. Ranking histórico das seleções
# =========================

print("\nTop 10 seleções por taxa de vitória histórica:")
print(
    stats.sort_values("taxa_vitoria", ascending=False)
    .head(10)
    [["team", "jogos", "media_gols_marcados", "media_gols_sofridos", "taxa_vitoria"]]
)


# =========================
# 8. Preencher planilha da Copa de 2026
# =========================

jogos_preenchidos, arquivo_saida, arquivo_previsoes = preencher_planilha_copa()

print(f"\nPlanilha preenchida: {arquivo_saida}")
print(f"Tabela simples de previsoes: {arquivo_previsoes}")
print(f"Jogos preenchidos: {jogos_preenchidos}")
