from .data import *
from .model import *
from .evaluation import *

def pipline_phase_I(input_countries_pretrain, output_countries_pretrain, output_countries_test, 
                    df_train, df_val, df_test, y_scalers, 
                    emb_dim, num_experts, epochs, batch_size, quantiles):
    
    print(f"\n=== Phase I: Pretraining ===")
    print(f"Input  Countries for Pretraining: {input_countries_pretrain}")
    print(f"Output Countries for Pretraining: {output_countries_pretrain}")
    print(f"Output Countries for Testing: {output_countries_test}")

    some_c = input_countries_pretrain[0]
    x1_shape, x2_shape, y_dim = df_train[some_c]["X_lag"].shape[1:], df_train[some_c]["X_lead"].shape[1:], df_train[some_c]["Y"].shape[1]  

    PhaseI = build_graph_gated_quantile_model(x1_shape=x1_shape, x2_shape=x2_shape, y_dim=y_dim, quantiles=quantiles, emb_dim=emb_dim, num_experts=num_experts)
    PhaseI.compile(optimizer="adam", loss=QuantileLoss(quantiles))

    gate_fn_p1 = lambda target: [target]  # naive: only itself
    X1tr, X2tr, Gtr, Ytr = pack_dataset(df_train, input_countries_pretrain, output_countries_pretrain, gate_fn_p1)
    X1va, X2va, Gva, Yva = pack_dataset(df_val,   input_countries_pretrain, output_countries_pretrain, gate_fn_p1)

    ckpt = f"Model/PhaseI_{output_countries_test[0]}.keras" if len(output_countries_test) == 1 else "Model/PhaseI_best.keras"
    PhaseI.fit({"X_lag_all": X1tr, "X_lead_all": X2tr, "graph_gate": Gtr}, Ytr,
               validation_data=({"X_lag_all": X1va, "X_lead_all": X2va, "graph_gate": Gva}, Yva), 
               epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[save_best_ckpt(ckpt)])

    PhaseI_best = load_model(ckpt)
    test_metrics = evaluate_countries(model=PhaseI_best, split=df_test, 
                                      input_countries=output_countries_test, output_countries=output_countries_test,
                                      gate_fn=gate_fn_p1, quantiles=quantiles, y_scalers=y_scalers)
    return test_metrics


def pipline_phase_II(input_countries, target_country, adjacency_dict, graph_degree, 
                     df_train, df_val, df_test, y_scalers, 
                     emb_dim, num_experts, epochs, batch_size,
                     quantiles=[0.1, 0.5, 0.9], model_choice="PhaseI_best"):
    
    print(f"\n=== Phase II: Fine-Tuning ===")
    print(f"Input Countries: {input_countries}")
    print(f"Target country: {target_country}")
    print(f"Graph degree: {graph_degree}")

    some_c = input_countries[0]
    x1_shape, x2_shape, y_dim = df_train[some_c]["X_lag"].shape[1:], df_train[some_c]["X_lead"].shape[1:], df_train[some_c]["Y"].shape[1]  

    # Load best Phase I, then Phase II fine-tune
    ckpt = f"Model/{model_choice}.keras"
    #ckpt = "Model/PhaseI_AT.keras"
    PhaseI_best = load_model(ckpt)
    PhaseII = build_graph_gated_quantile_model(x1_shape=x1_shape, x2_shape=x2_shape, y_dim=y_dim, quantiles=quantiles, emb_dim=emb_dim, num_experts=num_experts)
    PhaseII.set_weights(PhaseI_best.get_weights())
    # transfer weights
    #transfer_moe_weights(PhaseI_best, PhaseII, num_experts=num_experts)
    #PhaseII.get_layer("enc_gate").trainable = False
    #for e in range(num_experts):
    #    PhaseII.get_layer(f"enc_expert{e}").trainable = False

    PhaseII.compile(optimizer="adam", loss=QuantileLoss(quantiles))

    graph_knowledge=get_k_hop_countries(adjacency_dict, input_countries, target_country, graph_degree)
    gate_fn_p2 = lambda target: graph_knowledge  # target implicit via graph gate
    X1tr, X2tr, Gtr, Ytr = pack_dataset(df_train, input_countries, [target_country], gate_fn_p2)
    X1va, X2va, Gva, Yva = pack_dataset(df_val,   input_countries, [target_country], gate_fn_p2)

    val_before_full_shot = PhaseII.evaluate({"X_lag_all": X1va, "X_lead_all": X2va, "graph_gate": Gva}, Yva, batch_size=batch_size, verbose=0)
    print(val_before_full_shot)

    ckpt = f"Model/phase2_best_{target_country}_deg{graph_degree}.keras"
    PhaseII.fit({"X_lag_all": X1tr, "X_lead_all": X2tr, "graph_gate": Gtr}, Ytr,
                validation_data=({"X_lag_all": X1va, "X_lead_all": X2va, "graph_gate": Gva}, Yva),
                epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[save_best_ckpt(ckpt)])

    
    #X1te2, X2te2, Gte2, Yte2 = pack_dataset(df_test, input_countries, [target_country], gate_fn_p2)
    #pred = PhaseII_best.predict({"X_lag_all": X1te2, "X_lead_all": X2te2, "graph_gate": Gte2}, verbose=0)
    PhaseII_best = load_model(ckpt)
    test_metrics_p2 = evaluate_countries(model=PhaseII_best, split=df_test,                 
                                         input_countries=input_countries, output_countries=[target_country],
                                         gate_fn=gate_fn_p2, quantiles=quantiles, y_scalers=y_scalers,
    )
    return test_metrics_p2