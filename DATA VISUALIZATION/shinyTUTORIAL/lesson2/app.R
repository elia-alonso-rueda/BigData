#SHINY TUTORIAL 
###Lesson2. STRUCTURE AND GUI OF APPS DEVELOPED WITH SHINY
library(shiny)

# UI 
ui <- fluidPage(
  titlePanel("My Shiny App - LESSON 2"),
  sidebarLayout(
    sidebarPanel(
      h2("Installation"),
      p("Shiny is available on CRAN, so you can install it in the usual way from your R console:"),
      code('install.packages("shiny")'),
      br(),
      "Shiny is a product of ", 
      span("RStudio", style = "color:blue"),
      br(),
      img(src = "etsiinf.jpg", height = 100, width = 100),
      br(),
      "We are students from ", span("ETSIINF", style = "color:green")
    ),
    mainPanel(
      h1("Introducing Shiny"),
      p("Shiny is a new package from RStudio that makes it ", 
        em("incredibly easy "), 
        "to build interactive web applications with R."),
      br(),
      p("For an introduction and live examples, visit the ",
        a("Shiny homepage.", 
          href = "http://shiny.rstudio.com")),
      br(),
      h2("Features"),
      p("- Build useful web applications with only a few lines of code-no JavaScript required."),
      p("- Shiny applications are automatically 'live' in the same way that ", 
        strong("spreadsheets"),
        " are live. Outputs change instantly as users modify inputs, without requiring a reload of the browser.")
    )
  )
)

# server
server <- function(input, output) {
  
}

# Run the app ----
shinyApp(ui = ui, server = server)

