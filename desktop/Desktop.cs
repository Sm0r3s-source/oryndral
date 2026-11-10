using System;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Threading;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;
using Microsoft.Win32;

internal static class Desktop
{
    [STAThread]
    private static void Main()
    {
        new Application { ShutdownMode = ShutdownMode.OnMainWindowClose }.Run(new DesktopWindow());
    }
}

internal sealed class DesktopWindow : Window
{
    private static readonly Brush Page = BrushAt("#F0F4F9");
    private static readonly Brush Panel = Brushes.White;
    private static readonly Brush Text = BrushAt("#1F2937");
    private static readonly Brush Muted = BrushAt("#53617A");
    private static readonly Brush Line = BrushAt("#DCE3EF");
    private static readonly Brush Accent = BrushAt("#0B57D0");
    private static readonly Brush Good = BrushAt("#17764C");
    private static readonly Brush Bad = BrushAt("#A6253D");

    private readonly LocalFiles files = new LocalFiles();
    private readonly WebView2 chat = new WebView2();
    private readonly TextBox model = new TextBox();
    private readonly TextBlock modelName = new TextBlock();
    private readonly ComboBox device = new ComboBox();
    private readonly ComboBox context = new ComboBox();
    private readonly Button browse = new Button();
    private readonly Button start = new Button();
    private readonly Button stop = new Button();
    private readonly Button copy = new Button();
    private readonly TextBlock status = new TextBlock();
    private readonly TextBlock idleTitle = new TextBlock();
    private readonly TextBlock idleDescription = new TextBlock();
    private readonly Border idle = new Border();
    private readonly TextBox logs = new TextBox();
    private readonly Expander details = new Expander();
    private bool browserReady, closing;
    private ManagedServer server;
    private CancellationTokenSource startup;
    private Uri origin;

    public DesktopWindow()
    {
        Title = "Oryndral Desktop";
        Width = 1240;
        Height = 810;
        MinWidth = 840;
        MinHeight = 610;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        FontFamily = new FontFamily("Segoe UI");
        FontSize = 14;
        Foreground = Text;
        Background = Page;
        using (var icon = Assembly.GetExecutingAssembly().GetManifestResourceStream("Oryndral.ico"))
            if (icon != null)
                Icon = BitmapFrame.Create(icon, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        Content = Layout();
        Loaded += InitializeBrowser;
        Closing += CloseDesktop;
        ShowStatus("Preparing chat", "The chat view is starting.", Muted);
        UpdateControls();
    }

    private UIElement Layout()
    {
        var root = new Grid { Margin = new Thickness(22) };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(308) });
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var header = new DockPanel { Margin = new Thickness(0, 0, 0, 16) };
        var brand = new StackPanel();
        brand.Children.Add(new TextBlock { Text = "Oryndral", FontSize = 29, FontWeight = FontWeights.Bold });
        brand.Children.Add(new TextBlock { Text = "LOCAL AI DESKTOP", FontSize = 11, Foreground = Muted });
        status.HorizontalAlignment = HorizontalAlignment.Right;
        status.VerticalAlignment = VerticalAlignment.Center;
        status.FontWeight = FontWeights.SemiBold;
        DockPanel.SetDock(brand, Dock.Left);
        header.Children.Add(brand);
        header.Children.Add(status);
        Grid.SetColumnSpan(header, 2);
        root.Children.Add(header);

        var controls = new StackPanel();
        controls.Children.Add(new TextBlock { Text = "Model setup", FontSize = 18, FontWeight = FontWeights.SemiBold });
        controls.Children.Add(new TextBlock
        {
            Text = "Choose a local GGUF model, then start its server.",
            TextWrapping = TextWrapping.Wrap, Foreground = Muted, Margin = new Thickness(0, 5, 0, 0)
        });
        controls.Children.Add(Field("GGUF model"));
        model.Text = files.SuggestedModel ?? "";
        model.Padding = new Thickness(7);
        model.ToolTip = "Path to a local GGUF file";
        model.TextChanged += (_, __) => UpdateControls();
        AutomationProperties.SetName(model, "GGUF model path");
        controls.Children.Add(model);
        modelName.Foreground = Muted;
        modelName.FontSize = 12;
        modelName.TextTrimming = TextTrimming.CharacterEllipsis;
        modelName.Margin = new Thickness(0, 5, 0, 0);
        controls.Children.Add(modelName);
        SetButton(browse, "Browse models…");
        browse.Margin = new Thickness(0, 8, 0, 0);
        browse.HorizontalAlignment = HorizontalAlignment.Left;
        browse.Click += ChooseModel;
        controls.Children.Add(browse);

        controls.Children.Add(Field("Compute device"));
        device.Items.Add("CPU");
        device.Items.Add("NVIDIA CUDA");
        device.SelectedIndex = files.CudaReady ? 1 : 0;
        device.SelectionChanged += (_, __) => UpdateControls();
        AutomationProperties.SetName(device, "Compute device");
        controls.Children.Add(device);

        controls.Children.Add(Field("Context length"));
        foreach (var item in new[] { "Model default (up to 2,048)", "512", "1024", "2048", "4096", "8192" })
            context.Items.Add(item);
        context.SelectedIndex = 0;
        AutomationProperties.SetName(context, "Context length");
        controls.Children.Add(context);

        var buttons = new UniformGrid { Columns = 2, Margin = new Thickness(0, 16, 0, 0) };
        SetButton(start, "Start server", true);
        SetButton(stop, "Stop server");
        start.Margin = new Thickness(0, 0, 8, 0);
        start.Click += StartServer;
        stop.Click += (_, __) => StopServer();
        buttons.Children.Add(start);
        buttons.Children.Add(stop);
        controls.Children.Add(buttons);

        SetButton(copy, "Copy API address");
        copy.Margin = new Thickness(0, 8, 0, 0);
        copy.Click += CopyAddress;
        controls.Children.Add(copy);

        details.Header = "Server details";
        details.Margin = new Thickness(0, 15, 0, 0);
        logs.IsReadOnly = true;
        logs.FontFamily = new FontFamily("Consolas");
        logs.FontSize = 11;
        logs.TextWrapping = TextWrapping.Wrap;
        logs.Height = 144;
        logs.Padding = new Thickness(7);
        logs.VerticalScrollBarVisibility = ScrollBarVisibility.Auto;
        logs.Margin = new Thickness(0, 8, 0, 0);
        AutomationProperties.SetName(logs, "Server output");
        details.Content = logs;
        controls.Children.Add(details);
        var left = Card(new ScrollViewer { Content = controls, VerticalScrollBarVisibility = ScrollBarVisibility.Auto }, 16);
        left.Margin = new Thickness(0, 0, 14, 0);
        Grid.SetRow(left, 1);
        root.Children.Add(left);

        var view = new Grid();
        chat.Visibility = Visibility.Hidden;
        view.Children.Add(chat);
        idleTitle.FontSize = 23;
        idleTitle.FontWeight = FontWeights.SemiBold;
        idleDescription.TextWrapping = TextWrapping.Wrap;
        idleDescription.Foreground = Muted;
        idleDescription.Margin = new Thickness(0, 8, 0, 0);
        var emptyText = new StackPanel
        {
            MaxWidth = 400, VerticalAlignment = VerticalAlignment.Center,
            HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(24)
        };
        emptyText.Children.Add(idleTitle);
        emptyText.Children.Add(idleDescription);
        idle.Child = emptyText;
        idle.Background = Panel;
        view.Children.Add(idle);
        var right = Card(view, 0);
        Grid.SetRow(right, 1);
        Grid.SetColumn(right, 1);
        root.Children.Add(right);
        return root;
    }

    private async void InitializeBrowser(object sender, RoutedEventArgs args)
    {
        Loaded -= InitializeBrowser;
        try
        {
            var environment = await CoreWebView2Environment.CreateAsync(null, files.BrowserDataDirectory);
            if (closing) return;
            await chat.EnsureCoreWebView2Async(environment);
            if (closing) return;
            var web = chat.CoreWebView2;
            web.Settings.IsWebMessageEnabled = false;
            web.Settings.AreHostObjectsAllowed = false;
            web.NavigationStarting += (_, e) => e.Cancel = !IsLocalChat(e.Uri);
            web.NewWindowRequested += (_, e) => e.Handled = true;
            web.PermissionRequested += (_, e) => e.State = CoreWebView2PermissionState.Deny;
            web.DownloadStarting += (_, e) => e.Cancel = true;
            browserReady = true;
            ShowStatus("Ready to start", "Select a GGUF model and start the server to open a local chat.", Muted);
        }
        catch (Exception error)
        {
            if (closing) return;
            ShowStatus("Chat view unavailable", "The embedded chat could not start. See server details for the error.", Bad);
            AppendLog(error.Message);
            details.IsExpanded = true;
        }
        UpdateControls();
    }

    private bool IsLocalChat(string target)
    {
        Uri uri;
        return origin != null && Uri.TryCreate(target, UriKind.Absolute, out uri)
            && uri.Scheme == Uri.UriSchemeHttp && uri.Host == "127.0.0.1" && uri.Port == origin.Port;
    }

    private void ChooseModel(object sender, RoutedEventArgs args)
    {
        var picker = new OpenFileDialog
        {
            Title = "Select an Oryndral model", Filter = "GGUF models (*.gguf)|*.gguf", CheckFileExists = true
        };
        var current = model.Text.Trim();
        if (File.Exists(current))
        {
            picker.InitialDirectory = Path.GetDirectoryName(current);
            picker.FileName = Path.GetFileName(current);
        }
        if (picker.ShowDialog(this) == true)
            model.Text = picker.FileName;
    }

    private async void StartServer(object sender, RoutedEventArgs args)
    {
        if (server != null || !browserReady)
            return;
        var path = model.Text.Trim().Trim('"');
        if (!File.Exists(path) || !String.Equals(Path.GetExtension(path), ".gguf", StringComparison.OrdinalIgnoreCase))
        {
            ShowStatus("Choose a model", "Browse to an existing GGUF file.", Bad);
            return;
        }
        var useCuda = device.SelectedIndex == 1;
        var engine = useCuda ? files.CudaEngine : files.CpuEngine;
        if (engine == null || (useCuda && !files.CudaReady))
        {
            ShowStatus("Engine unavailable", useCuda
                ? "The CUDA engine and NVIDIA runtime libraries are required. You can choose CPU to run locally."
                : "Place the CPU Oryndral executable in the desktop app's engine folder.", Bad);
            return;
        }
        int size;
        int? contextLength = Int32.TryParse(context.SelectedItem as string, out size) ? size : (int?)null;
        logs.Clear();
        var current = new ManagedServer(engine, Path.GetFullPath(path), useCuda ? files.CudaBin : null, contextLength);
        var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(3));
        server = current;
        startup = cancellation;
        current.Output += line => Dispatcher.BeginInvoke(new Action(() =>
        {
            if (server == current) AppendLog(line);
        }));
        current.Exited += _ => Dispatcher.BeginInvoke(new Action(() => EngineExited(current)));
        ShowStatus("Starting server", "Oryndral is loading " + Path.GetFileName(path) + ".", Muted);
        UpdateControls();
        try
        {
            Uri ready = await current.StartAsync(cancellation.Token);
            if (server != current || closing) return;
            origin = ready;
            chat.Source = ready;
            idle.Visibility = Visibility.Collapsed;
            chat.Visibility = Visibility.Visible;
            ShowStatus("Server running · " + (useCuda ? "CUDA" : "CPU"), "", Good);
        }
        catch (Exception error)
        {
            if (server != current || closing) return;
            AppendLog(error is OperationCanceledException ? "Model startup exceeded three minutes." : error.Message);
            StopServer("Server could not start", "See server details for the model or startup error.", Bad);
            details.IsExpanded = true;
        }
        finally
        {
            if (startup == cancellation) startup = null;
            cancellation.Dispose();
            UpdateControls();
        }
    }

    private void EngineExited(ManagedServer current)
    {
        if (server != current || closing) return;
        StopServer("Server stopped", "The engine exited. See server details for more information.", Bad);
        details.IsExpanded = true;
    }

    private void StopServer(string title = "Server stopped", string description = "Start a model to begin another chat.", Brush color = null)
    {
        var running = server;
        server = null;
        startup?.Cancel();
        running?.Dispose();
        origin = null;
        chat.Visibility = Visibility.Hidden;
        idle.Visibility = Visibility.Visible;
        ShowStatus(title, description, color ?? Muted);
        UpdateControls();
    }

    private void CloseDesktop(object sender, CancelEventArgs args)
    {
        closing = true;
        StopServer();
        chat.Dispose();
    }

    private void CopyAddress(object sender, RoutedEventArgs args)
    {
        if (origin == null) return;
        try
        {
            Clipboard.SetText(new Uri(origin, "v1").AbsoluteUri);
            status.Text = "API address copied";
        }
        catch (System.Runtime.InteropServices.ExternalException)
        {
            AppendLog("The clipboard is busy. API address: " + new Uri(origin, "v1").AbsoluteUri);
            details.IsExpanded = true;
        }
    }

    private void ShowStatus(string title, string description, Brush color)
    {
        status.Text = title;
        status.Foreground = color;
        idleTitle.Text = title;
        idleDescription.Text = description;
    }

    private void UpdateControls()
    {
        var path = model.Text.Trim().Trim('"');
        modelName.Text = File.Exists(path) ? Path.GetFileName(path) : "";
        modelName.Visibility = String.IsNullOrEmpty(modelName.Text) ? Visibility.Collapsed : Visibility.Visible;
        var available = server == null && !closing;
        start.IsEnabled = browserReady && available && !String.IsNullOrWhiteSpace(model.Text);
        stop.IsEnabled = server != null && !closing;
        copy.IsEnabled = origin != null && !closing;
        model.IsEnabled = browse.IsEnabled = device.IsEnabled = context.IsEnabled = available;
    }

    private void AppendLog(string value)
    {
        if (logs.Text.Length > 24000)
            logs.Text = logs.Text.Substring(logs.Text.Length - 12000);
        logs.AppendText(value + Environment.NewLine);
        logs.ScrollToEnd();
    }

    private static TextBlock Field(string name)
    {
        return new TextBlock
        {
            Text = name, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 16, 0, 5)
        };
    }

    private static void SetButton(Button button, string name, bool primary = false)
    {
        button.Content = name;
        button.Padding = new Thickness(10, 6, 10, 6);
        button.Background = primary ? Accent : Panel;
        button.Foreground = primary ? Brushes.White : Text;
        button.BorderBrush = primary ? Accent : Line;
        button.BorderThickness = new Thickness(1);
        button.MinHeight = 34;
        var frame = new FrameworkElementFactory(typeof(Border), "Frame");
        frame.SetValue(Border.BackgroundProperty, new TemplateBindingExtension(Control.BackgroundProperty));
        frame.SetValue(Border.BorderBrushProperty, new TemplateBindingExtension(Control.BorderBrushProperty));
        frame.SetValue(Border.BorderThicknessProperty, new TemplateBindingExtension(Control.BorderThicknessProperty));
        frame.SetValue(Border.CornerRadiusProperty, new CornerRadius(16));
        var content = new FrameworkElementFactory(typeof(ContentPresenter));
        content.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        content.SetValue(ContentPresenter.VerticalAlignmentProperty, VerticalAlignment.Center);
        content.SetValue(ContentPresenter.MarginProperty, new TemplateBindingExtension(Control.PaddingProperty));
        frame.AppendChild(content);
        var template = new ControlTemplate(typeof(Button)) { VisualTree = frame };
        foreach (var state in new[] { Button.IsMouseOverProperty, Button.IsPressedProperty })
        {
            var trigger = new Trigger { Property = state, Value = true };
            trigger.Setters.Add(new Setter(UIElement.OpacityProperty, 0.82, "Frame"));
            template.Triggers.Add(trigger);
        }
        var disabled = new Trigger { Property = Button.IsEnabledProperty, Value = false };
        disabled.Setters.Add(new Setter(UIElement.OpacityProperty, 0.48, "Frame"));
        template.Triggers.Add(disabled);
        button.Template = template;
    }

    private static Border Card(UIElement child, int padding)
    {
        return new Border
        {
            Child = child, Background = Panel, BorderBrush = Line,
            BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(10), Padding = new Thickness(padding)
        };
    }

    private static Brush BrushAt(string value)
    {
        return (Brush)new BrushConverter().ConvertFromString(value);
    }

}
