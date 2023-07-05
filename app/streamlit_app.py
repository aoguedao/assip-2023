import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.integrate import solve_ivp

sns.set_theme(style="whitegrid")


def sird_forward(
    t_array,
    y0,
    N,
    beta,
    gamma,
    mu
):

    def func(t, y):
        S, I, R, D = y
        dS_dt = - beta * S / N * I
        dI_dt = beta * S / N * I - gamma * I - mu * I
        dR_dt = gamma * I
        dD_dt = mu * I
        return np.array([dS_dt, dI_dt, dR_dt, dD_dt])

    t_span = (t_array[0], t_array[-1])
    sol = solve_ivp(func, t_span, y0, t_eval=t_array)
    return sol.y.T


st.set_page_config(
     page_title="SEIRD Model",
     page_icon="🧬",
     layout="wide",
)

st.title("SIRD Model")

tab_sird, tab_help = st.tabs(["SIRD", "Help"])

with tab_sird:

    st.latex(r"""
        \begin{align*}
        \frac{dS}{dt} &= - \frac{\beta}{N}  S I \\
        \frac{dI}{dt} &= \frac{\beta}{N} S I - \gamma  I - \mu I \\
        \frac{dR}{dt} &= \gamma I \\
        \frac{dD}{dt} &= \mu I \\
        \end{align*}
    """
    )

    col11, col12, col13, col14, col15 = st.columns(5)

    N = col11.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    n_days = col12.number_input(
        "Number of days",
        min_value=30,
        max_value=366,
        value=60,
        step=1,
        format=None,
        help="Number of days for simulation",
    )

    beta = col13.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.001,
        format=None,
        help="Transmission Rate",
    )

    gamma = col14.number_input(
        "Gamma",
        min_value=0.001,
        max_value=1.0,
        value=1 / 14,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Recovered",
    )

    mu = col15.number_input(
        "Mu",
        min_value=0.001,
        max_value=1.0,
        value=1 / 5,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Dead",
    )


    if st.button("Run forward simulation"):
        with st.spinner('Wait for it...'):
            # Initial conditions
            t_train = np.arange(0, n_days, 1)
            parameters = {
                "beta": beta,
                "gamma": gamma,
                "mu": mu,
            }
            S_0 = N - 1
            I_0 = 1
            R_0 = 0
            D_0 = 0
            y0 = [S_0, I_0, R_0, D_0]
            y_sol = sird_forward(t_train, y0, N, beta, gamma, mu)

            # Create dataframe
            model_name = "SIRD"
            populations_names = list(model_name)
            data_real = (
                    pd.DataFrame(y_sol, columns=populations_names)
                    .assign(time=t_train)
                    .melt(id_vars="time", var_name="status", value_name="population")
            )

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                data=data_real,
                x="time",
                y="population",
                hue="status",
                legend=True,
                linestyle="dashed",
                ax=ax
            )
            ax.set_title(f"{model_name} model - Forward Simulation")
            st.pyplot(fig)


with tab_help:
    url = "https://raw.githubusercontent.com/aoguedao/neural-computing-book/main/images/coyoya.jpg"
    st.header("Work in progress... Here is a kitty.")
    st.image(url, width=200)

