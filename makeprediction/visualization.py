
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from makeprediction.gpts import IGaussianProcessTimeSerie
from makeprediction.exceptions import NotValidModelError
class IVisualizer(ABC):
    # @abstractmethod
    # def plot():
    #     pass

    @abstractmethod
    def iplot():
        pass

class Visualizer(IVisualizer):
    # def __init__(self,model:IGaussianProcessTimeSerie):
    #     self.model = model
    @classmethod
    def qgpplot(cls,model:IGaussianProcessTimeSerie, template="plotly_white", return_fig=False, train_only=False,
                test_only=False,
                data_mode='lines',
                model_mode='lines',
                prediction_mode='lines'):
        '''Plot function of the gp model using plotly.'''
        if not isinstance(model, IGaussianProcessTimeSerie):
            raise NotValidModelError(f'{model}: is not a valid model.')

        if all([test_only, train_only]):
            print('when both arguments: "test_only", "train_only" are True, None is returned.')
            return go.Figure()

        if model._yfit is None:
            x_list = model._xtrain.tolist()
            # x_rev = x_list[::-1]
            y_list = model._ytrain.tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_list, y=y_list,
                line_color='rgba(0,0,0, 1)',
                name='Training data', showlegend=True,
                mode=data_mode,
            ))
            fig.update_layout(template=template)
            fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right", x=1))
            fig.update_yaxes(automargin=True)

            if return_fig:
                return fig
            else:
                fig.show()
        elif (model._xtest is None) | (train_only):
            # training plot
            x_list = model._xtrain.tolist()
            # x_rev = x_list[::-1]
            y_list = model._ytrain.tolist()
            yf_list = model._yfit.tolist()
            y_upper = model._yfit + 1.96 * model._std_yfit
            y_upper = y_upper.tolist()
            y_lower = model._yfit - 1.96 * model._std_yfit
            y_lower = y_lower.tolist()
            # y_lower_rev = y_lower[::-1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_list, y=y_list,
                line_color='rgba(0,0,0, 1)',
                name='Training data', mode=data_mode,
            ))

            fig.add_trace(go.Scatter(
                x=x_list, y=yf_list,
                line_color='rgba(255,0,0, 1)',
                name='Model',
                mode=model_mode,

            ))

            # fig.update_traces(mode='lines')
            fig.update_layout(template=template)
            fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right", x=1))
            fig.update_yaxes(automargin=True)

            if return_fig:
                return fig
            else:
                fig.show()

        else:
            if test_only:
                # prediction plot
                xs_list = model._xtest.tolist()
                xs_rev = xs_list[::-1]
                yp_list = model._ypred.tolist()
                yp_upper = model._ypred + 1.96 * model._std_ypred
                yp_upper = yp_upper.tolist()
                yp_lower = model._ypred - 1.96 * model._std_ypred
                yp_lower = yp_lower.tolist()
                yp_lower_rev = yp_lower[::-1]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=model._xtest, y=yp_list,
                    line_color='rgba(0,0,255, .8)',
                    name='Prediction', mode=prediction_mode))
                fig.add_trace(go.Scatter(
                    x=xs_list + xs_rev,
                    y=yp_upper + yp_lower_rev,
                    fill='toself',
                    fillcolor='rgba(0,0,255,.1)',
                    line_color='rgba(255,255,255,0)',
                    name='Prediction confidence interval(95%)',
                    showlegend=True))
                # fig.update_traces(mode='lines')
                fig.update_layout(template=template)
                fig.update_layout(legend=dict(orientation="h",
                                              yanchor="bottom",
                                              y=1.02,
                                              xanchor="right", x=1))
                fig.update_yaxes(automargin=True)

            else:
                # prediction plot
                xs_list = model._xtest.tolist()
                xs_rev = xs_list[::-1]
                yp_list = model._ypred.tolist()
                yp_upper = model._ypred + 1.96 * model._std_ypred
                yp_upper = yp_upper.tolist()
                yp_lower = model._ypred - 1.96 * model._std_ypred
                yp_lower = yp_lower.tolist()
                yp_lower_rev = yp_lower[::-1]
                # training plot
                x_list = model._xtrain.tolist()
                # x_rev = x_list[::-1]
                y_list = model._ytrain.tolist()
                yf_list = model._yfit.tolist()
                y_upper = model._yfit + 1.96 * model._std_yfit
                y_upper = y_upper.tolist()
                y_lower = model._yfit - 1.96 * model._std_yfit
                y_lower = y_lower.tolist()
                # y_lower_rev = y_lower[::-1]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_list, y=y_list,
                    line_color='rgba(0,0,0, 1)',
                    name='Training data', mode=data_mode,
                ))
                fig.add_trace(go.Scatter(
                    x=x_list, y=yf_list,
                    line_color='rgba(255,0,0, 1)',
                    name='Model',
                    mode=model_mode,
                ))
                fig.add_trace(go.Scatter(
                    x=model._xtest, y=yp_list,
                    line_color='rgba(0,0,255, .8)',
                    name='Prediction', mode=prediction_mode))
                fig.add_trace(go.Scatter(
                    x=xs_list + xs_rev,
                    y=yp_upper + yp_lower_rev,
                    fill='toself',
                    fillcolor='rgba(0,0,255,.1)',
                    line_color='rgba(255,255,255,0)',
                    name='Prediction confidence interval(95%)',
                    showlegend=True))
                # fig.update_traces(mode='lines')
                fig.update_layout(template=template)
                fig.update_layout(legend=dict(orientation="h",
                                              yanchor="bottom",
                                              y=1.02,
                                              xanchor="right", x=1))
                fig.update_yaxes(automargin=True)

            if return_fig:
                return fig
            else:
                fig.show()
    @classmethod
    def iplot(cls,model, x_test=None, y_test=None, template=None,
               return_fig=False, train_only=False, test_only=False,
               data_mode='lines',
               model_mode='lines',
               prediction_mode='lines',
               new_data_mode='lines'):
        '''Plot function of the gaussian process model and prediction using plotly package.'''

        if all([x_test is None, y_test is None]):
            fig = cls.qgpplot(model, return_fig=True, train_only=train_only,
                               test_only=test_only, data_mode=data_mode,
                               model_mode=model_mode,
                               prediction_mode=prediction_mode,
                               )
            if return_fig:
                return fig
            return fig.show()

        elif x_test is None:
            fig = cls.qgpplot(model, return_fig=True, train_only=train_only,
                               test_only=test_only, data_mode=data_mode,
                               model_mode=model_mode, prediction_mode=prediction_mode,
                               )
            print('warrning: the input x_test is null.')
            if return_fig:
                return fig
            return fig.show()

        elif y_test is None:
            fig = cls.qgpplot(model, return_fig=True, train_only=train_only,
                               test_only=test_only, data_mode=data_mode,
                               model_mode=model_mode, prediction_mode=prediction_mode,)
            print('warrning: the input y_test is null.')
            if return_fig:
                return fig
            return fig.show()

        else:
            # fig = go.Figure()

            fig = cls.qgpplot(model, return_fig=True, train_only=train_only,
                               test_only=test_only, data_mode=data_mode,
                               model_mode=model_mode, prediction_mode=prediction_mode,
                               )
            fig.add_trace(go.Scatter(x=x_test, y=y_test, line_color='rgba(128,128,128, .7)',
                                     name='Testing data', mode=new_data_mode))
            # fig.update_traces(mode='lines')
            fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right", x=1))
            fig.update_yaxes(automargin=True)
            if return_fig:
                return fig
            return fig.show()

    @classmethod
    def iplot_components(cls,
            model,
            save=False,
            filename=None,
            template="plotly_white"):

        kernels_list = model.kernel.label().split(" + ")
        kernels_list = kernels_list + ["Noise"]
        kernels_list = [f"{kernels_list[i]}:  {i+1}-th component." for i in range(len(kernels_list))]
        if len(kernels_list) == 2:
            fig = make_subplots(
                rows=len(kernels_list),
                cols=1,
                subplot_titles=kernels_list)
            fig.append_trace(go.Scatter(
                x=model._xtrain,
                y=model._yfit), row=1, col=1)
            fig.append_trace(go.Scatter(
                x=model._xtrain,
                y=model._ytrain - model._yfit),
                row=2, col=1)
            fig.update_layout(template=template)
            fig.update_layout(showlegend=False)
            fig.update_yaxes(automargin=True)
            fig.show()

        else:
            fig = make_subplots(
                rows=len(kernels_list),
                cols=1,
                subplot_titles=kernels_list)
            for i in range(len(kernels_list)):
                if i < len(model.components):
                    fig.append_trace(go.Scatter(
                        x=model._xtrain,
                        y=model.components[i]), row=i + 1, col=1)
                else:
                    fig.append_trace(go.Scatter(
                        x=model._xtrain,
                        y=model._ytrain - model._yfit),
                        row=i + 1, col=1)
            # fig.update_layout(legend= {'itemsizing': 'constant'})
            # fig.update_layout(height=700, width=900)
            fig.update_layout(template=template)
            fig.update_layout(showlegend=False)
            fig.update_yaxes(automargin=True)

            fig.show()
        if save:
            # fig.html
            if filename is None:
                filename = "fig"

            fig.write_html(filename + ".html")

    