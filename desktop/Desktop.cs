using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Interop;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
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

// Colors match the chat page (src/web_chat.html) and follow the Windows app theme.
internal static class Theme
{
    public static readonly bool Dark = SystemUsesDarkApps();
    public static readonly Brush Page = Pick("#FAF9FD", "#171722");
    public static readonly Brush Panel = Pick("#FFFFFF", "#202130");
    public static readonly Brush Soft = Pick("#EEEDF9", "#292B43");
    public static readonly Brush Line = Pick("#DCDCED", "#3A3D55");
    public static readonly Brush Text = Pick("#232538", "#F0EFF7");
    public static readonly Brush Muted = Pick("#5C6075", "#B6B8CC");
    public static readonly Brush Accent = Pick("#4657AA", "#BBC5FF");
    public static readonly Brush AccentText = Pick("#FFFFFF", "#202B62");
    public static readonly Brush Good = Pick("#34694F", "#9CD5B5");
    public static readonly Brush Bad = Pick("#A22B45", "#FFA8BA");

    private const string Xaml = @"
<ResourceDictionary xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation'
                    xmlns:x='http://schemas.microsoft.com/winfx/2006/xaml'>
  <Style TargetType='Button'>
    <Setter Property='Foreground' Value='{DynamicResource Text}'/>
    <Setter Property='Background' Value='{DynamicResource Panel}'/>
    <Setter Property='BorderBrush' Value='{DynamicResource Line}'/>
    <Setter Property='BorderThickness' Value='1'/>
    <Setter Property='Padding' Value='14,7'/>
    <Setter Property='MinHeight' Value='36'/>
    <Setter Property='Cursor' Value='Hand'/>
    <Setter Property='FocusVisualStyle' Value='{x:Null}'/>
    <Setter Property='Template'>
      <Setter.Value>
        <ControlTemplate TargetType='Button'>
          <Border x:Name='Frame' CornerRadius='18' Background='{TemplateBinding Background}'
                  BorderBrush='{TemplateBinding BorderBrush}' BorderThickness='{TemplateBinding BorderThickness}'>
            <ContentPresenter HorizontalAlignment='Center' VerticalAlignment='Center' Margin='{TemplateBinding Padding}'/>
          </Border>
          <ControlTemplate.Triggers>
            <Trigger Property='IsMouseOver' Value='True'><Setter TargetName='Frame' Property='Opacity' Value='0.84'/></Trigger>
            <Trigger Property='IsKeyboardFocused' Value='True'><Setter TargetName='Frame' Property='BorderBrush' Value='{DynamicResource Accent}'/></Trigger>
            <Trigger Property='IsEnabled' Value='False'><Setter TargetName='Frame' Property='Opacity' Value='0.45'/></Trigger>
          </ControlTemplate.Triggers>
        </ControlTemplate>
      </Setter.Value>
    </Setter>
  </Style>
  <Style TargetType='RadioButton'>
    <Setter Property='Foreground' Value='{DynamicResource Muted}'/>
    <Setter Property='Cursor' Value='Hand'/>
    <Setter Property='FocusVisualStyle' Value='{x:Null}'/>
    <Setter Property='Template'>
      <Setter.Value>
        <ControlTemplate TargetType='RadioButton'>
          <Border x:Name='Frame' CornerRadius='13' Padding='2,6' Background='Transparent'
                  BorderBrush='Transparent' BorderThickness='1'>
            <ContentPresenter HorizontalAlignment='Center' VerticalAlignment='Center'/>
          </Border>
          <ControlTemplate.Triggers>
            <Trigger Property='IsMouseOver' Value='True'><Setter Property='Foreground' Value='{DynamicResource Text}'/></Trigger>
            <Trigger Property='IsChecked' Value='True'>
              <Setter TargetName='Frame' Property='Background' Value='{DynamicResource Soft}'/>
              <Setter Property='Foreground' Value='{DynamicResource Text}'/>
              <Setter Property='FontWeight' Value='SemiBold'/>
            </Trigger>
            <Trigger Property='IsKeyboardFocused' Value='True'><Setter TargetName='Frame' Property='BorderBrush' Value='{DynamicResource Accent}'/></Trigger>
            <Trigger Property='IsEnabled' Value='False'><Setter TargetName='Frame' Property='Opacity' Value='0.45'/></Trigger>
          </ControlTemplate.Triggers>
        </ControlTemplate>
      </Setter.Value>
    </Setter>
  </Style>
  <Style TargetType='ScrollBar'>
    <Setter Property='Width' Value='10'/>
    <Setter Property='MinWidth' Value='10'/>
    <Setter Property='Template'>
      <Setter.Value>
        <ControlTemplate TargetType='ScrollBar'>
          <Grid Background='Transparent'>
            <Track x:Name='PART_Track' IsDirectionReversed='True'>
              <Track.Thumb>
                <Thumb>
                  <Thumb.Template>
                    <ControlTemplate TargetType='Thumb'>
                      <Border Margin='3,2' CornerRadius='2' Background='{DynamicResource Line}'/>
                    </ControlTemplate>
                  </Thumb.Template>
                </Thumb>
              </Track.Thumb>
            </Track>
          </Grid>
        </ControlTemplate>
      </Setter.Value>
    </Setter>
  </Style>
</ResourceDictionary>";

    public static ResourceDictionary Styles()
    {
        var styles = (ResourceDictionary)XamlReader.Parse(Xaml);
        styles["Panel"] = Panel;
        styles["Soft"] = Soft;
        styles["Line"] = Line;
        styles["Text"] = Text;
        styles["Muted"] = Muted;
        styles["Accent"] = Accent;
        return styles;
    }

    private static Brush Pick(string light, string dark)
    {
        var brush = (Brush)new BrushConverter().ConvertFromString(Dark ? dark : light);
        brush.Freeze();
        return brush;
    }

    private static bool SystemUsesDarkApps()
    {
        try
        {
            return Registry.GetValue(@"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                "AppsUseLightTheme", 1) is int light && light == 0;
        }
        catch (Exception error) when (error is IOException || error is System.Security.SecurityException)
        {
            return false;
        }
    }
}

internal sealed class DesktopWindow : Window
{
    private readonly LocalFiles files = new LocalFiles();
    private readonly WebView2 chat = new WebView2();
    private readonly System.Windows.Shapes.Ellipse statusDot = new System.Windows.Shapes.Ellipse();
    private readonly TextBlock status = new TextBlock();
    private readonly TextBlock modelName = new TextBlock();
    private readonly TextBlock modelDetail = new TextBlock();
    private readonly TextBlock idleTitle = new TextBlock();
    private readonly TextBlock idleDescription = new TextBlock();
    private readonly RadioButton cpu = Option("CPU");
    private readonly RadioButton gpu = Option("NVIDIA GPU");
    private readonly RadioButton[] contexts =
    {
        Option("Auto"), Option("512", 512), Option("1K", 1024), Option("2K", 2048), Option("4K", 4096), Option("8K", 8192)
    };
    private readonly Button browse = new Button();
    private readonly Button action = new Button();
    private readonly Button copy = new Button();
    private readonly Button logToggle = new Button();
    private readonly TextBox logs = new TextBox();
    private readonly Border idle = new Border();
    private Border logFrame;
    private string modelPath, lastLine;
    private bool browserReady, closing;
    private ManagedServer server;
    private CancellationTokenSource startup;
    private Uri origin;

    public DesktopWindow()
    {
        Title = "Oryndral Desktop";
        Width = 1180;
        Height = 800;
        MinWidth = 860;
        MinHeight = 600;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        FontFamily = new FontFamily("Segoe UI Variable Text, Segoe UI");
        FontSize = 13.5;
        Foreground = Theme.Text;
        Background = Theme.Page;
        UseLayoutRounding = true;
        Resources = Theme.Styles();
        ImageSource mark = null;
        using (var icon = Assembly.GetExecutingAssembly().GetManifestResourceStream("Oryndral.ico"))
            if (icon != null)
            {
                var frames = new IconBitmapDecoder(icon, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames;
                Icon = mark = frames[0];
                foreach (var frame in frames)
                    if (frame.PixelWidth > ((BitmapSource)mark).PixelWidth)
                        mark = frame;
            }
        modelPath = files.SuggestedModel;
        (files.CudaReady ? gpu : cpu).IsChecked = true;
        contexts[0].IsChecked = true;
        Content = Layout(mark);
        Loaded += InitializeBrowser;
        Closing += CloseDesktop;
        ShowStatus("Preparing", "The chat view is starting.", Theme.Muted);
        UpdateControls();
    }

    protected override void OnSourceInitialized(EventArgs args)
    {
        base.OnSourceInitialized(args);
        // Older Windows versions ignore this attribute and keep the standard title bar.
        int dark = Theme.Dark ? 1 : 0;
        DwmSetWindowAttribute(new WindowInteropHelper(this).Handle, 20, ref dark, sizeof(int));
    }

    private UIElement Layout(ImageSource mark)
    {
        var setup = new StackPanel { Margin = new Thickness(20, 20, 20, 12) };
        statusDot.Width = statusDot.Height = 9;
        statusDot.Margin = new Thickness(1, 2, 9, 0);
        statusDot.VerticalAlignment = VerticalAlignment.Center;
        status.FontSize = 15;
        status.FontWeight = FontWeights.SemiBold;
        status.TextTrimming = TextTrimming.CharacterEllipsis;
        var state = new DockPanel();
        DockPanel.SetDock(statusDot, Dock.Left);
        state.Children.Add(statusDot);
        state.Children.Add(status);
        setup.Children.Add(state);

        setup.Children.Add(Heading("Model"));
        modelName.FontWeight = FontWeights.SemiBold;
        modelName.TextTrimming = TextTrimming.CharacterEllipsis;
        modelDetail.FontSize = 12;
        modelDetail.Foreground = Theme.Muted;
        modelDetail.TextTrimming = TextTrimming.CharacterEllipsis;
        modelDetail.Margin = new Thickness(0, 3, 0, 0);
        var model = new StackPanel();
        model.Children.Add(modelName);
        model.Children.Add(modelDetail);
        setup.Children.Add(Well(model, 12));
        browse.Content = "Choose model…";
        browse.Margin = new Thickness(0, 8, 0, 0);
        browse.Click += ChooseModel;
        setup.Children.Add(browse);

        setup.Children.Add(Heading("Run on"));
        setup.Children.Add(Segments("Compute device", cpu, gpu));
        if (!files.CudaReady)
        {
            gpu.ToolTip = "The CUDA engine or the NVIDIA runtime files were not found.";
            ToolTipService.SetShowOnDisabled(gpu, true);
        }

        setup.Children.Add(Heading("Context length"));
        setup.Children.Add(Segments("Context length", contexts));
        setup.Children.Add(new TextBlock
        {
            Text = "Auto uses the model's own limit, up to 2,048 tokens.",
            FontSize = 12, Foreground = Theme.Muted, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(2, 7, 0, 0)
        });

        action.MinHeight = 40;
        action.FontWeight = FontWeights.SemiBold;
        action.Margin = new Thickness(0, 24, 0, 0);
        action.Click += (_, __) => { if (server == null) StartServer(); else StopServer(); };
        setup.Children.Add(action);
        copy.Content = "Copy API address";
        copy.Margin = new Thickness(0, 8, 0, 0);
        copy.ToolTip = "Address for other OpenAI-compatible apps";
        copy.Click += CopyAddress;
        setup.Children.Add(copy);

        logToggle.Background = logToggle.BorderBrush = Brushes.Transparent;
        logToggle.Foreground = Theme.Muted;
        logToggle.FontSize = 12;
        logToggle.Padding = new Thickness(8, 4, 8, 4);
        logToggle.MinHeight = 0;
        logToggle.HorizontalAlignment = HorizontalAlignment.Left;
        logToggle.Click += (_, __) => ShowLog(logFrame.Visibility != Visibility.Visible);
        logs.IsReadOnly = true;
        logs.FontFamily = new FontFamily("Cascadia Mono, Consolas");
        logs.FontSize = 11;
        logs.TextWrapping = TextWrapping.Wrap;
        logs.Height = 150;
        logs.Padding = new Thickness(8);
        logs.BorderThickness = new Thickness(0);
        logs.Background = Brushes.Transparent;
        logs.Foreground = Theme.Muted;
        logs.VerticalScrollBarVisibility = ScrollBarVisibility.Auto;
        AutomationProperties.SetName(logs, "Server log");
        logFrame = Well(logs, 0);
        logFrame.Margin = new Thickness(0, 4, 0, 0);
        var footer = new StackPanel { Margin = new Thickness(20, 0, 20, 14) };
        footer.Children.Add(logToggle);
        footer.Children.Add(logFrame);
        ShowLog(false);

        var side = new DockPanel();
        DockPanel.SetDock(footer, Dock.Bottom);
        side.Children.Add(footer);
        side.Children.Add(new ScrollViewer { Content = setup, VerticalScrollBarVisibility = ScrollBarVisibility.Auto });

        idleTitle.FontFamily = new FontFamily("Palatino Linotype, Georgia");
        idleTitle.FontSize = 32;
        idleTitle.TextAlignment = TextAlignment.Center;
        idleTitle.TextWrapping = TextWrapping.Wrap;
        idleTitle.Margin = new Thickness(0, 18, 0, 0);
        idleDescription.Foreground = Theme.Muted;
        idleDescription.TextAlignment = TextAlignment.Center;
        idleDescription.TextWrapping = TextWrapping.Wrap;
        idleDescription.Margin = new Thickness(0, 10, 0, 0);
        var message = new StackPanel
        {
            MaxWidth = 460, Margin = new Thickness(24),
            VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center
        };
        message.Children.Add(new Image { Source = mark, Width = 52, Height = 52 });
        message.Children.Add(idleTitle);
        message.Children.Add(idleDescription);
        idle.Child = message;
        idle.Background = Theme.Page;
        chat.Visibility = Visibility.Hidden;
        var view = new Grid();
        view.Children.Add(chat);
        view.Children.Add(idle);

        var root = new Grid();
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(304) });
        root.ColumnDefinitions.Add(new ColumnDefinition());
        root.Children.Add(new Border
        {
            Child = side, Background = Theme.Panel, BorderBrush = Theme.Line, BorderThickness = new Thickness(0, 0, 1, 0)
        });
        Grid.SetColumn(view, 1);
        root.Children.Add(view);
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
            web.Profile.PreferredColorScheme = Theme.Dark
                ? CoreWebView2PreferredColorScheme.Dark : CoreWebView2PreferredColorScheme.Light;
            web.Settings.IsWebMessageEnabled = false;
            web.Settings.AreHostObjectsAllowed = false;
            web.NavigationStarting += (_, e) => e.Cancel = !IsLocalChat(e.Uri);
            web.NewWindowRequested += OpenLink;
            web.PermissionRequested += (_, e) => e.State = CoreWebView2PermissionState.Deny;
            web.DownloadStarting += (_, e) => e.Cancel = true;
            browserReady = true;
            ShowStatus("Not running", "Choose a model and start the server to begin chatting.", Theme.Muted);
        }
        catch (Exception error)
        {
            if (closing) return;
            AppendLog(error.Message);
            ShowStatus("Chat view unavailable", error.Message, Theme.Bad);
            ShowLog(true);
        }
        UpdateControls();
    }

    private bool IsLocalChat(string target)
    {
        Uri uri;
        return origin != null && Uri.TryCreate(target, UriKind.Absolute, out uri)
            && uri.Scheme == Uri.UriSchemeHttp && uri.Host == "127.0.0.1" && uri.Port == origin.Port;
    }

    // Links in a reply open in the default browser. The chat view itself never leaves the local server.
    private static void OpenLink(object sender, CoreWebView2NewWindowRequestedEventArgs args)
    {
        args.Handled = true;
        Uri link;
        if (!Uri.TryCreate(args.Uri, UriKind.Absolute, out link) || link.IsLoopback
            || (link.Scheme != Uri.UriSchemeHttp && link.Scheme != Uri.UriSchemeHttps))
            return;
        try { Process.Start(new ProcessStartInfo(link.AbsoluteUri) { UseShellExecute = true }); }
        catch (Win32Exception) { }
    }

    private void ChooseModel(object sender, RoutedEventArgs args)
    {
        var picker = new OpenFileDialog
        {
            Title = "Choose a model", Filter = "GGUF models (*.gguf)|*.gguf", CheckFileExists = true
        };
        if (File.Exists(modelPath))
        {
            picker.InitialDirectory = Path.GetDirectoryName(modelPath);
            picker.FileName = Path.GetFileName(modelPath);
        }
        if (picker.ShowDialog(this) == true)
            modelPath = picker.FileName;
        UpdateControls();
    }

    private async void StartServer()
    {
        if (server != null || !browserReady || !File.Exists(modelPath))
            return;
        var useCuda = gpu.IsChecked == true;
        var engine = useCuda ? files.CudaEngine : files.CpuEngine;
        if (engine == null || (useCuda && !files.CudaReady))
        {
            ShowStatus("Engine not found", useCuda
                ? "The CUDA engine and NVIDIA runtime files are required. Choose CPU to run without them."
                : "Place the CPU Oryndral executable in the app's engine folder.", Theme.Bad);
            return;
        }
        int? contextLength = null;
        foreach (var option in contexts)
            if (option.IsChecked == true)
                contextLength = (int?)option.Tag;
        logs.Clear();
        lastLine = null;
        var current = new ManagedServer(engine, Path.GetFullPath(modelPath), useCuda ? files.CudaBin : null, contextLength);
        var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(3));
        server = current;
        startup = cancellation;
        current.Output += line => Dispatcher.BeginInvoke(new Action(() =>
        {
            if (server == current) AppendLog(line);
        }));
        current.Exited += _ => Dispatcher.BeginInvoke(new Action(() =>
        {
            if (server == current && !closing) Fail("Server stopped", lastLine ?? "The engine exited.");
        }));
        ShowStatus("Starting", "Loading " + Path.GetFileNameWithoutExtension(modelPath)
            + ". Large models can take a minute.", Theme.Accent);
        UpdateControls();
        try
        {
            Uri ready = await current.StartAsync(cancellation.Token);
            if (server != current || closing) return;
            origin = ready;
            chat.Source = ready;
            idle.Visibility = Visibility.Collapsed;
            chat.Visibility = Visibility.Visible;
            ShowStatus("Running on " + (useCuda ? "GPU" : "CPU"), "", Theme.Good);
        }
        catch (Exception error)
        {
            if (server != current || closing) return;
            Fail("Could not start", error is OperationCanceledException
                ? "The model took longer than three minutes to load." : lastLine ?? error.Message);
        }
        finally
        {
            if (startup == cancellation) startup = null;
            cancellation.Dispose();
            UpdateControls();
        }
    }

    private void Fail(string title, string reason)
    {
        StopServer(title, reason, Theme.Bad);
        ShowLog(true);
    }

    private void StopServer(string title = "Not running",
        string description = "Start the server to begin another chat.", Brush color = null)
    {
        var running = server;
        server = null;
        startup?.Cancel();
        running?.Dispose();
        origin = null;
        chat.Visibility = Visibility.Hidden;
        idle.Visibility = Visibility.Visible;
        ShowStatus(title, description, color ?? Theme.Muted);
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
        var address = new Uri(origin, "v1").AbsoluteUri;
        try
        {
            Clipboard.SetText(address);
            copy.Content = "Copied";
            var reset = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1.5) };
            reset.Tick += (_, __) =>
            {
                reset.Stop();
                copy.Content = "Copy API address";
            };
            reset.Start();
        }
        catch (ExternalException)
        {
            AppendLog("The clipboard is busy. API address: " + address);
            ShowLog(true);
        }
    }

    private void ShowStatus(string title, string description, Brush color)
    {
        status.Text = title;
        statusDot.Fill = color;
        idleTitle.Text = title;
        idleDescription.Text = description;
    }

    private void ShowLog(bool open)
    {
        logFrame.Visibility = open ? Visibility.Visible : Visibility.Collapsed;
        logToggle.Content = (open ? "▾  " : "▸  ") + "Server log";
    }

    private void UpdateControls()
    {
        var found = File.Exists(modelPath);
        modelName.Text = found ? Path.GetFileNameWithoutExtension(modelPath) : "No model selected";
        modelDetail.Text = found
            ? FileSize(new FileInfo(modelPath).Length) + " · " + Path.GetDirectoryName(modelPath)
            : "Choose a GGUF file to begin.";
        modelDetail.ToolTip = found ? modelPath : null;

        var stopped = server == null;
        action.Content = stopped ? "Start server" : origin == null ? "Cancel" : "Stop server";
        action.IsEnabled = !closing && (!stopped || (browserReady && found));
        action.Background = action.BorderBrush = stopped ? Theme.Accent : Theme.Panel;
        action.Foreground = stopped ? Theme.AccentText : Theme.Text;
        if (!stopped) action.BorderBrush = Theme.Line;
        copy.Visibility = origin != null ? Visibility.Visible : Visibility.Collapsed;

        var editable = stopped && !closing;
        browse.IsEnabled = cpu.IsEnabled = editable;
        gpu.IsEnabled = editable && files.CudaReady;
        foreach (var option in contexts)
            option.IsEnabled = editable;
    }

    private void AppendLog(string value)
    {
        lastLine = value;
        if (logs.Text.Length > 24000)
            logs.Text = logs.Text.Substring(logs.Text.Length - 12000);
        logs.AppendText(value + Environment.NewLine);
        logs.ScrollToEnd();
    }

    private static string FileSize(long bytes)
    {
        return bytes >= 1L << 30
            ? (bytes / (double)(1L << 30)).ToString("0.0") + " GB"
            : (bytes / (double)(1L << 20)).ToString("0") + " MB";
    }

    private static TextBlock Heading(string name)
    {
        return new TextBlock
        {
            Text = name.ToUpperInvariant(), FontSize = 11, FontWeight = FontWeights.SemiBold,
            Foreground = Theme.Muted, Margin = new Thickness(2, 22, 0, 7)
        };
    }

    private static RadioButton Option(string name, int? tokens = null)
    {
        return new RadioButton { Content = name, Tag = tokens, FontSize = 12.5 };
    }

    private static Border Segments(string name, params RadioButton[] options)
    {
        var row = new UniformGrid { Rows = 1 };
        foreach (var option in options)
        {
            option.GroupName = name;
            row.Children.Add(option);
        }
        AutomationProperties.SetName(row, name);
        return Well(row, 3, 16);
    }

    private static Border Well(UIElement child, double padding, double radius = 12)
    {
        return new Border
        {
            Child = child, Background = Theme.Page, BorderBrush = Theme.Line, BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(radius), Padding = new Thickness(padding)
        };
    }

    [DllImport("dwmapi.dll")]
    private static extern int DwmSetWindowAttribute(IntPtr window, int attribute, ref int value, int size);
}
