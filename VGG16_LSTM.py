import tensorflow as tf
from keras.models import Sequential
#from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN, Reshape, TimeDistributed
from keras import regularizers, optimizers
import pandas as pd
#from keras_preprocessing.image import DataFrameIterator
from TimeDistributedImageDataGenerator import TimeDistributedImageDataGenerator
#from tf.keras.models import load_model
from tensorflow.keras.models import load_model
#from tensorflow.python.debug.examples.v1.debug_keras import tf
#from tensorflow.python.keras.optimizers import Adam

#import tensorflow as tf
#from keras_preprocessing.image import ImageDataGenerator
#from keras_preprocessing.image import array_to_img, img_to_array, load_img
#from keras_preprocessing.image import DirectoryIterator, DataFrameIterator, os
#import numpy as np


df=pd.read_csv('/media/proactionlab/Storage/NeuralNet_80Tools/AttractorModel/allLabels_shortTrainer.csv')
columns=["agrupado_em_porta-chaves", "associado_a_absorção", "associado_a_absorção_de_líquidos",
         "associado_a_ar_fresco", "associado_a_ar_quente", "associado_a_arte", "associado_a_baldes_do_lixo",
         "associado_à_boca", "associado_a_bolas/círculos", "associado_a_bolos", "associado_a_borrachas", "associado_a_bruxas",
         "associado_a_buracos", "associado_a_buracos_cilindricos", "associado_a_calor", "associado_a_candeeiros", "associado_a_carne",
         "associado_a_carros", "associado_a_cirurgia", "associado_a_claras_em_castelo", "associado_a_cofres", "associado_a_colher_de_café",
         "associado_a_colher_de_chá", "associado_a_colher_de_pau", "associado_a_colher_de_sobremesa", "associado_a_comida", "associado_a_compras",
         "associado_à_cor_verde", "associado_a_cores", "associado_a_detalhes", "associado_a_doenças", "associado_a_dor", "associado_a_dossiers",
         "associado_a_drogas", "associado_a_equipas", "associado_a_erros", "associado_à_escrita", "associado_a_espuma", "associado_a_espuma_de_barbear",
         "associado_a_estética", "associado_a_estradas", "associado_a_fadas", "associado_a_fechaduras", "associado_a_festejos", "associado_a_golfe", "associado_a_grafite",
         "associado_a_gramas", "associado_a_higiene", "associado_a_higiene_oral", "associado_a_informática", "associado_a_injecções", "associado_a_lápis",
         "associado_a_letras", "associado_a_limão", "associado_à_língua", "associado_a_magia", "associado_a_máquinas_de_costura", "associado_a_martelo",
         "associado_a_medicina", "associado_a_minas", "associado_a_músculos", "associado_a_passadeiras", "associado_a_peixe", "associado_a_pessoas_altas",
         "associado_a_pizza", "associado_a_portas", "associado_a_quilogramas", "associado_a_saladas", "associado_a_sangue", "associado_à_saúde", "associado_a_segurança",
         "associado_à_Suiça", "associado_a_sujidade", "associado_a_sumo", "associado_a_trabalhos_manuais", "associado_a_um_alvo", "associado_a_um_cursor", "associado_a_um_trabalho",
         "associado_a_vácuo", "associado_a_vento", "associado_a_verniz_das_unhas", "associado_a_xadrez", "associado_ao_bico_do_lápis", "associado_ao_olho", "associado_ao_perigo",
         "associado_ao_sabor", "associado_ao_tabaco", "associado_com_o_frio", "associado_com_o_sopro(desopro)", "associado_com_pontaria", "asssociado_a_lâmpadas", "contém_ar",
         "cruzeta", "dá_faísca", "é_aberto_no_topo", "é_afiado", "é_afunilado", "é_afunilado_em_cima", "é_alto", "é_amarelo", "é_antigo", "é_apagável", "é_arredondado", "é_azul",
         "é_barulhento", "é_branco", "é_castanho", "é_cinzento", "é_côncavo", "é_cor_de_laranja", "é_descartável", "é_desdobrável", "é_dobrado", "é_duro", "é_eléctrica", "é_electrónico",
         "é_específico", "é_esponjoso", "é_fino", "é_flexível", "é_fundo", "é_grande", "é_inflamável", "é_insuflável", "é_leve", "é_loiça", "é_longo", "é_macio", "é_mole", "é_pequeno",
         "é_perigoso", "é_pesado", "é_pontiagudo/tem_ponta_aguçada", "é_portátil", "é_preso_à_parede", "é_preto", "é_quadrado", "é_quebrável", "é_quente", "é_reciclável", "é_rectangular",
         "é_rugoso", "é_semelhante_a_uma_pá_na_extremidade", "é_transparente", "é_transparente", "é_triangular", "é_um_brinquedo", "é_uma_arma", "é_uma_máquina", "é_verde", "é_vermelho",
         "encaixa_na_carica", "encaixa_na_fechadura", "encaixa_na_mão", "encontra-se_dentro_da_caixa_de_fósforos", "encontra-se_em_armários", "encontra-se_em_bares",
         "encontra-se_em_cabeleireiros", "encontra-se_em_cafés", "encontra-se_em_caixas_de_costura", "encontra-se_em_caixas_de_ferramentas", "encontra-se_em_casas_de_banho",
         "encontra-se_em_centros_de_saúde", "encontra-se_em_cozinhas", "encontra-se_em_escolas", "encontra-se_em_escritórios", "encontra-se_em_estojos", "encontra-se_em_gavetas",
         "encontra-se_em_ginásios", "encontra-se_em_hospitais", "encontra-se_em_igrejas", "encontra-se_em_laboratórios", "encontra-se_em_lojas", "encontra-se_em_mesas",
         "encontra-se_em_oficinas", "encontra-se_em_papelarias", "encontra-se_em_pastelarias", "encontra-se_em_piscinas", "encontra-se_em_salas_de_aula", "encontra-se_em_salas_de_estar",
         "encontra-se_em_secretárias", "encontra-se_em_supermercados", "encontra-se_nas_bicicletas", "encontra-se_nas_casas", "encontra-se_nas_portas", "encontra-se_no_saco_de_golfe",
         "encontra-se_nos_bolsos", "encontra-se_nos_carros", "está_incorporado_no_computador", "está_ligado_ao_computador", "faz_um_som_agudo", "feito_de_aço", "feito_de_arame",
         "feito_de_barro", "feito_de_borracha", "feito_de_cerâmica", "feito_de_cortiça", "feito_de_cristal", "feito_de_diferentes_materiais", "feito_de_esponja", "feito_de_feltro",
         "feito_de_ferro", "feito_de_grafite", "feito_de_madeira", "feito_de_metal", "feito_de_papel", "feito_de_plástico", "feito_de_porcelana", "feito_de_tecido", "feito_de_vidro",
         "magoa", "multiplica_a_força_aplicada", "parece_o_formato_de_uma_pistola", "parece_um_palito", "parece_um_vaso", "parece_uma_lança", "parece_uma_lixa", "parece_uma_seta",
         "pode_não_ter_um_fio", "pode_não_ter_um_recipiente", "pode_ser_táctil/touchpad", "produz_aparas", "produz_dois_buracos", "produz_dois_buracos_de_cada_vez", "produz_lixo",
         "queima", "rasga-se", "requer_ar", "requer_força", "requer_ser_afiado", "requer_ser_espremido", "requer_uma_moeda", "salta", "serve_para_abanar", "serve_para_abrir",
         "serve_para_abrir_cadeados", "serve_para_abrir_fechaduras", "serve_para_abrir_garrafas", "serve_para_abrir_portas", "serve_para_acampar", "serve_para_acender",
         "serve_para_acender", "serve_para_acender_cigarros", "serve_para_acender_o_fogão", "serve_para_acender_velas", "serve_para_acertar_num_alvo", "serve_para_afiar",
         "serve_para_afiar_lápis", "serve_para_agarrar", "serve_para_agarrar_comida", "serve_para_agrafar", "serve_para_agrafar_papeis", "serve_para_alisar/achatar",
         "serve_para_alisar/achatar_a_massa", "serve_para_amassar", "serve_para_ampliar", "serve_para_apagar", "serve_para_apagar_escrita", "serve_para_apagar_giz",
         "serve_para_apagar_grafite", "serve_para_apagar_lápis", "serve_para_apagar_marcador", "serve_para_apagar_quadros", "serve_para_apanhar", "serve_para_apanhar_lixo",
         "serve_para_apertar", "serve_para_apertar_parafusos", "serve_para_apertar_porcas", "serve_para_armazenar", "serve_para_armazenar_comida", "serve_para_armazenar_flores",
         "serve_para_armazenar_líquidos", "serve_para_arrancar", "serve_para_arrancar_pregos", "serve_para_atear_fogo", "serve_para_barbear", "serve_para_bater", "serve_para_bater_a_bola",
         "serve_para_bater_claras", "serve_para_bater_ingredientes", "serve_para_beber", "serve_para_beber_líquidos", "serve_para_borrifar", "serve_para_borrifar_água",
         "serve_para_buzinar", "serve_para_carimbar", "serve_para_carregar", "serve_para_carregar_compras", "serve_para_cavar", "serve_para_cavar_a_terra", "serve_para_cavar_buracos",
         "serve_para_cavar_buracos_na_terra", "serve_para_certificar", "serve_para_comer", "serve_para_comer_cereais", "serve_para_comer_comida_liquida", "serve_para_comer_sopa",
         "serve_para_conduzir", "serve_para_controlar", "serve_para_corrigir", "serve_para_cortar", "serve_para_cortar_as_unhas", "serve_para_cortar_comida", "serve_para_cortar_paper",
         "serve_para_coser", "serve_para_coser_roupas", "serve_para_cozinhar", "serve_para_cozinhar_sopa", "serve_para_cultivar", "serve_para_decoração", "serve_para_decorar_bolos",
         "serve_para_depilação", "serve_para_desapertar", "serve_para_descascar", "serve_para_descascar_comida", "serve_para_descascar_fruta", "serve_para_descascar_vegetais",
         "serve_para_desembraçar_o_cabelo", "serve_para_desenhar", "serve_para_desentupir", "serve_para_desentupir_canos", "serve_para_desentupir_sanitas", "serve_para_diminuir_o_calor",
         "serve_para_dobrar", "serve_para_driblar", "serve_para_empurrar_a_água", "serve_para_encher_copos", "serve_para_escovar", "serve_para_escovar_os_dentes", "serve_para_escrever",
         "serve_para_esguichar", "serve_para_espalhar", "serve_para_espetar", "serve_para_espetar_comida", "serve_para_espremer", "serve_para_espremer_citrinos",
         "serve_para_espremer_fruta", "serve_para_esticar", "serve_para_esticar_a_massa", "serve_para_estudar", "serve_para_extrair_o_sumo", "serve_para_fazer_bicos_no_lápis",
         "serve_para_fazer_buracos", "serve_para_fazer_buracos_na_madeira", "serve_para_fazer_buracos_nas_paredes", "serve_para_fazer_buracos_no_papel", "serve_para_fazer_exercício",
         "serve_para_fazer_fogo", "serve_para_fazer_musculação", "serve_para_fechar", "serve_para_fechar_garrafas", "serve_para_fechar_garrafas_de_vinho", "serve_para_fechar_portas",
         "serve_para_fechar_sacos", "serve_para_fixar", "serve_para_flutuar", "serve_para_flutuar_na_água", "serve_para_fumar", "serve_para_gravar", "serve_para_iluminar",
         "serve_para_injectar", "serve_para_investigar", "serve_para_jogar", "serve_para_jogar_basquetebol", "serve_para_jogar_golfe", "serve_para_jogar_ténis", "serve_para_lançar",
         "serve_para_lançar_bolas", "serve_para_lavar", "serve_para_lavar_a_loiça", "serve_para_lavar_o_chão", "serve_para_lavar_o_corpo", "serve_para_lavar_os_dentes",
         "serve_para_levar_comida_à_boca", "serve_para_limar", "serve_para_limar_unhas", "serve_para_limpar", "serve_para_limpar_a_boca", "serve_para_limpar_as_mãos",
         "serve_para_limpar_o_chão", "serve_para_maquilhar", "serve_para_marcar", "serve_para_marcar_com_tinta", "serve_para_marcar_papeis", "serve_para_marcar_papeis_com_tinta",
         "serve_para_martelar", "serve_para_martelar_pregos", "serve_para_misturar", "serve_para_misturar_comida", "serve_para_movimentar", "serve_para_movimentar_a_seta/cursor",
         "serve_para_movimentar_barcos", "serve_para_nadar", "serve_para_não_afogar", "serve_para_não_afundar_na_água", "serve_para_não_entornar", "serve_para_organizar",
         "serve_para_pendurar", "serve_para_pendurar_malas", "serve_para_pendurar_roupas", "serve_para_pentear/escovar", "serve_para_pentear/escovar_os_cabelos", "serve_para_pesar",
         "serve_para_pesar_na_balança", "serve_para_pescar", "serve_para_pintar", "serve_para_pintar_quadros", "serve_para_plantar", "serve_para_practicar_desporto",
         "serve_para_praticar_lançamento_de_peso", "serve_para_produzir_ar_frio", "serve_para_produzir_ar_quente", "serve_para_produzir_vento", "serve_para_produzir/fabricar",
         "serve_para_produzir/fabricar_pneus", "serve_para_proteger", "serve_para_puxar_o_fio_de_pesca", "serve_para_quebrar/partir", "serve_para_quebrar/partir_a_casca",
         "serve_para_quebrar/partir_a_casca_das_nozes", "serve_para_quebrar/partir_nozes", "serve_para_ralar", "serve_para_ralar_comida", "serve_para_ralar_queijo", "serve_para_rechear",
         "serve_para_reduzir_o_tamanho", "serve_para_reduzir_o_tamanho_das_unhas", "serve_para_refrescar", "serve_para_regar", "serve_para_regar_flores", "serve_para_remar",
         "serve_para_reparar", "serve_para_retirar", "serve_para_retirar_as_caricas", "serve_para_retirar_pêlos", "serve_para_rodar", "serve_para_rodar_parafusos",
         "serve_para_rodar_porcas", "serve_para_salvar(vidas)", "serve_para_secar", "serve_para_secar_o_cabelo", "serve_para_secar_roupa", "serve_para_segurar", "serve_para_segurar_velas",
         "serve_para_seleccionar", "serve_para_servir", "serve_para_sinalizar", "serve_para_temperar", "serve_para_tirar_sangue", "serve_para_tocar_na_bola", "serve_para_trabalhar",
         "serve_para_transformar_em_pedaços_pequenos", "serve_para_transportar", "serve_para_transportar_compras", "serve_para_transportar_líquidos", "serve_para_tratar/arranjar",
         "serve_para_tratar/arranjar_as_unhas", "serve_para_tricotar", "serve_para_triturar", "serve_para_triturar_comida", "serve_para_triturar_pimenta", "serve_para_triturar_sopa",
         "serve_para_unir/juntar", "serve_para_unir/juntar_os_papeis", "serve_para_varrer", "serve_para_varrer_o_chão", "serve_para_vedar", "serve_para_ver",
         "serve_para_ver_coisas_pequenas", "serve_para_ver_detalhes", "serve_para_ver_mais_próximo", "serve_para_ver_melhor", "tem_a_forma_de_concha", "tem_bicos_com_várias_formas",
         "tem_botões", "tem_buracos", "tem_cabos", "tem_cabos_de_plástico", "tem_cerdas", "tem_cerdas_de_plástico", "tem_cerdas_sintéticas", "tem_cordas", "tem_decorações", "tem_dentes",
         "tem_desenhos", "tem_dois_botões", "tem_dois_buracos/argolas", "tem_dois_buracos/argolas_para_colocar_os_dedos", "tem_dois_cabos", "tem_dois_cabos_ligados", "tem_duas_lâminas",
         "tem_duas_varetas", "tem_ferrugem", "tem_forma_de_meia_esfera", "tem_forma_semicircular", "tem_gás", "tem_lâminas", "tem_luz", "tem_picos", "tem_pilhas", "tem_pó",
         "tem_pontas_arredondas", "tem_pontas_bicudas", "tem_pontuação", "tem_quatro_dentes", "tem_quatro_rodas", "tem_ranhura_para_a_moeda", "tem_riscas", "tem_riscas_pretas",
         "tem_rodas", "tem_rótulos", "tem_serrilha", "tem_símbolos", "tem_tinta", "tem_três_dentes", "tem_um_balão", "tem_um_bico", "tem_um_bico_de_plástico", "tem_um_botão",
         "tem_um_buraco", "tem_um_buraco_no_meio/centro", "tem_um_buraco_para_colocar_o_lápis", "tem_um_buraco_pequeno", "tem_um_cabo", "tem_um_cabo_de_madeira", "tem_um_cabo_de_metal",
         "tem_um_cabo_de_plástico", "tem_um_cabo_longo", "tem_um_cabo_para_empurrar", "tem_um_carreto", "tem_um_cesto", "tem_um_êmbolo", "tem_um_fio", "tem_um_fio_de_pesca",
         "tem_um_gancho", "tem_um_gargalo", "tem_um_gatilho", "tem_um_isco", "tem_um_motor", "tem_um_pé", "tem_um_recipiente", "tem_um_recipiente_para_o_sumo", "tem_um_saco",
         "tem_um_spray", "tem_um_vidro", "tem_uma_agulha", "tem_uma_almofada", "tem_uma_bateria", "tem_uma_bola", "tem_uma_bola_dentro", "tem_uma_cabeça_com_duas_pontas_diferentes",
         "tem_uma_cabeça_redonda", "tem_uma_cabeça/ponta", "tem_uma_cabeça/ponta_que_acende", "tem_uma_cabeça/ponta_vermelha", "tem_uma_cadeira/assento",
         "tem_uma_cadeira/assento_para_crianças", "tem_uma_cana", "tem_uma_chama", "tem_uma_corda", "tem_uma_escala", "tem_uma_escova_na_extremidade", "tem_uma_extremidade_ajustável",
         "tem_uma_extremidade_com_tiras_de_tecido", "tem_uma_extremidade_côncava", "tem_uma_faca", "tem_uma_forma_cónica", "tem_uma_forma_espiral", "tem_uma_lâmina", "tem_uma_lente",
         "tem_uma_lente_de_vidro", "tem_uma_lente_que_amplia", "tem_uma_lente_redonda", "tem_uma_lima", "tem_uma_luz", "tem_uma_luz_vermelha", "tem_uma_manivela", "tem_uma_mola",
         "tem_uma_mola_de_metal", "tem_uma_parte_com_duas_hastes/gancho", "tem_uma_parte_redonda", "tem_uma_parte_rugosa", "tem_uma_pedra", "tem_uma_ponta", "tem_uma_ponta_achatada",
         "tem_uma_ponta_côncava", "tem_uma_ponta_cónica", "tem_uma_ponta_de_borracha", "tem_uma_ponta_de_metal", "tem_uma_ranhura", "tem_uma_rede", "tem_uma_roda", "tem_uma_roldana",
         "tem_uma_rolha", "tem_uma_rosca", "tem_uma_tampa", "tem_uma_tesoura", "tem_uma_ventoinha", "tem_uma_ventosa", "tem_uma_zona_para_colocar_a_noz",
         "tem_uma_zona_para_encaixar_a_fruta", "tem_varetas", "tem_várias_peças_incorporadas", "tem_vários_filamentos", "um_electrodoméstico", "um_elemento_quimico", "um_filme",
         "um_objecto", "um_objecto_de_agricultura", "um_objecto_de_corte", "um_objecto_de_costura", "um_objecto_de_cozinha", "um_objecto_de_culinária", "um_objecto_de_decoração",
         "um_objecto_de_escritório", "um_objecto_de_estética", "um_objecto_de_lazer", "um_objecto_de_limpeza", "um_objecto_de_pintura", "um_objecto_de_sopro", "um_objecto_desportivo",
         "um_objecto_doméstico", "um_objecto_escolar", "um_objecto_médico", "um_recipiente", "um_talher", "uma_arma", "uma_máquina", "uma_massa", "uma_medida", "uma_peça",
         "uma_peça_de_xadrez/peão", "uma_pessoa", "uma_pessoa_que_anda_a_pé", "usa-se_à_volta_da_cintura", "usa-se_ao_abrir", "usa-se_ao_apertar/pressionar", "usa-se_ao_clicar",
         "usa-se_ao_empurrar", "usa-se_ao_empurrar_para_baixo", "usa-se_ao_encher", "usa-se_ao_encher_com_ar", "usa-se_ao_enfiar/colocar_os_dedos", "usa-se_ao_esfregar",
         "usa-se_ao_fechar", "usa-se_ao_mover_o_objecto", "usa-se_ao_pressionar_para_baixo", "usa-se_ao_pressionar/empurrar", "usa-se_ao_puxar", "usa-se_ao_raspar_o_fósforo",
         "usa-se_ao_raspar_o_fósforo_na_caixa", "usa-se_ao_rodar", "usa-se_ao_rodar_a_cabeça_e_o_corpo_em_sentidos_opostos", "usa-se_ao_soprar", "usa-se_com_a_boca",
         "usa-se_com_os_braços", "usa-se_com_um_pires", "usa-se_com_uma_pá", "usa-se_em_marcadores", "usa-se_na_sopa", "usa-se_na_sujidade", "usa-se_nas_paredes", "usa-se_no_giz",
         "usa-se_no_leite", "usa-se_no_pó", "usa-se_nos_cereais", "usa-se_por_sucção", "usa-se/agarra-se_com_as_mãos", "usado_à_noite", "usado_ao_lançar/atirar",
         "usado_ao_lançar/atirar_a_um_alvo", "usado_com_a_esfregona", "usado_com_a_vassoura", "usado_com_afia-lápis", "usado_com_agrafos", "usado_com_água", "usado_com_bolas",
         "usado_com_bolas_de_golfe", "usado_com_chave-de-fendas", "usado_com_copos", "usado_com_detergentes", "usado_com_farinha", "usado_com_papel", "usado_com_pasta_de_dentes",
         "usado_com_perfume", "usado_com_saca-rolhas", "usado_com_tinta", "usado_com_um_balde", "usado_com_um_cesto", "usado_com_um_computador", "usado_com_um_ecrã", "usado_com_um_garfo",
         "usado_com_um_martelo", "usado_com_um_tapete", "usado_com_uma_colher", "usado_com_uma_faca", "usado_com_uma_linha", "usado_com_uma_pena/volante", "usado_com_velas",
         "usado_durante_refeições", "usado_em_barcos", "usado_em_batatas", "usado_em_cadeados", "usado_em_canoas", "usado_em_canos", "usado_em_caricas", "usado_em_cenouras",
         "usado_em_cerveja", "usado_em_cigarros", "usado_em_citrinos", "usado_em_coisas_pequenas", "usado_em_cordas", "usado_em_desenhos", "usado_em_diferentes_veículos",
         "usado_em_empresas", "usado_em_especiarias", "usado_em_estendais", "usado_em_fechaduras", "usado_em_flores", "usado_em_fogueiras", "usado_em_frutas", "usado_em_garrafas",
         "usado_em_garrafas_de_vidro", "usado_em_jogos", "usado_em_jogos_de_futebol", "usado_em_laranjas", "usado_em_lareiras", "usado_em_lava-loiças", "usado_em_lugares_escuros",
         "usado_em_medicamentos", "usado_em_nozes", "usado_em_parafusos", "usado_em_pêlos", "usado_em_porcas", "usado_em_praias", "usado_em_pregos", "usado_em_quadros",
         "usado_em_quadros_pretos", "usado_em_queijo", "usado_em_salões_de_beleza", "usado_em_sanitas", "usado_em_seringas", "usado_em_tecidos", "usado_em_vacinas", "usado_em_vegetais",
         "usado_em_velas", "usado_na_agricultura", "usado_na_água", "usado_na_areia", "usado_na_broca", "usado_na_cara", "usado_na_carne", "usado_na_carpintaria", "usado_na_casca",
         "usado_na_horta", "usado_na_loiça", "usado_na_madeira", "usado_na_massa", "usado_na_pimenta", "usado_na_relva", "usado_na_terra", "usado_nas_balanças", "usado_nas_barbas",
         "usado_nas_malas", "usado_nas_mãos", "usado_nas_motas", "usado_nas_natas", "usado_nas_panelas", "usado_nas_roupas", "usado_nas_sobrancelhas", "usado_nas_unhas",
         "usado_no_badminton", "usado_no_banho", "usado_no_basquetebol", "usado_no_berbequim", "usado_no_cabelo", "usado_no_campo", "usado_no_campo_de_basquetebol",
         "usado_no_campo_de_golfe", "usado_no_chá", "usado_no_chão", "usado_no_corpo", "usado_no_desporto", "usado_no_fogão", "usado_no_jardim", "usado_no_jogo_das_setas",
         "usado_no_leite", "usado_no_lixo", "usado_no_mar", "usado_no_molhado/coisas_molhadas", "usado_no_peixe", "usado_no_pingue-pongue", "usado_no_ténis", "usado_no_trânsito",
         "usado_nos_casacos", "usado_nos_dentes", "usado_nos_lápis", "usado_nos_ovos", "usado_nos_pés", "usado_nos_rios", "usado_para_água", "usado_para_alertar", "usado_para_apitar",
         "usado_para_bebidas", "usado_para_bebidas_quentes", "usado_para_café", "usado_para_comida", "usado_para_espetar", "usado_para_fazer_barulho", "usado_para_lazer",
         "usado_para_líquidos", "usado_para_manicure", "usado_para_marcar_cestos", "usado_para_moldar_a_forma", "usado_para_pedicure", "usado_para_pressionar_para_baixo",
         "usado_para_sumo", "usado_para_vinho", "usado_pelo_movimento_de_abrir_e_fechar", "usado_pelos_polícias", "usado_por_agricultores", "usado_por_árbitos", "usado_por_artistas",
         "usado_por_carpinteiros", "usado_por_cientistas", "usado_por_costureiras", "usado_por_crianças", "usado_por_dentistas", "usado_por_detectives", "usado_por_enfermeiros",
         "usado_por_esteticistas", "usado_por_fumadores", "usado_por_homens", "usado_por_mágicos", "usado_por_médicos", "usado_por_mulheres", "usado_por_nadadores_salvadores",
         "usado_por_pasteleiros", "usado_por_pescadores", "usado_por_pintores", "usado_por_professores", "usado_quando_está_molhado", "usado_quando_está_molhado_em_tinta", "uso_diário",
         "utilizado_em_desportos_aquáticos", "utilizado_na_canoagem", "utilizado_na_construção_civil/obras", "várias_cores", "várias_formas", "várias_funções",
         "vários_níveis_de_potência", "vários_pesos", "vários_tamanhos", "Latada-festadosestudantes", "prego_no_prato"]

#img_width, img_height = 224, 224
#train_data_dir = 'db/train'
#validation_data_dir = 'db/test'
#nb_train_samples = 1800
#nb_validation_samples = 100
#num_timesteps = 10 # length of sequence
num_class = 855
batch_size = 10

datagen = TimeDistributedImageDataGenerator(time_steps = 20)
#datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=TimeDistributedImageDataGenerator(time_steps = 20)
train_generator=datagen.flow_from_dataframe(

#dataframe=df[:132160],
dataframe=df[:64000],
directory="/media/proactionlab/Storage/NeuralNet_80Tools/AttractorModel/images/train",
x_col="Filenames",
y_col=columns,
color_mode='rgb',
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=df[64000:76160],
directory="/media/proactionlab/Storage/NeuralNet_80Tools/AttractorModel/images/valid",
x_col="Filenames",
y_col=columns,
color_mode='rgb',
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

test_generator=datagen.flow_from_dataframe(
dataframe=df[76162:],
directory="/media/proactionlab/Storage/NeuralNet_80Tools/AttractorModel/images/test",
x_col="Filenames",
color_mode='rgb',
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))


#base_model = load_model('vgg16_80Tools.h5')
#base_model.summary()

# remove the last dense softmax layer which could output 1000 classes probability to
# include a 2 nodes dense layer
#model_vgg = Sequential()
#for layer in base_model.layers[:-1]:
 #   model_vgg.add(layer)

#model_vgg.summary()

# Building the model
model = Sequential()
model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(None,20, 224, 224, 3))))
model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(units=4096,activation="relu")))
model.add(TimeDistributed(Dense(units=4096,activation="relu")))
#model.add(TimeDistributed(model_vgg))
model.add(SimpleRNN(units=855, activation="sigmoid"))
model.compile(optimizers.Adam(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

imgs, labels = next(train_generator)
print(imgs.shape)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator),
                    epochs=45,
)
#model.fit(x=train_generator,
#          steps_per_epoch=len(train_generator),
#          validation_data=valid_generator,
#          validation_steps=len(valid_generator),
#          epochs=15,
#          verbose=2
#          )

model.summary()

model.save("vgg16_attractor_80tools.h5")
print("Saved model to disk")



