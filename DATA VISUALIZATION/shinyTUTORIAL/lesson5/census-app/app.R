#if you have never used these packages before, install them:
#install.packages(c("maps", "mapproj"))

# library(maps)
# library(mapproj)
# source("C:/Users/Elia/Desktop/master/BIG DATA/DATA VISUALIZATION/census-app/helpers.R")
# counties <- readRDS("C:/Users/Elia/Desktop/master/BIG DATA/DATA VISUALIZATION/census-app/data/counties.rds")
# percent_map(counties$white, "darkgreen", "% White")

#Important observations about shiny performance:
#Source scripts, load libraries, and read data sets at the beginning of app.R outside of the server function.
#Define user specific objects inside server function, but outside of any render* calls. 
#Only place code that Shiny must rerun to build an object inside of a render* function. Shiny will rerun all of the code in a render* chunk each time a user changes a widget mentioned in the chunk. This can be quite often.
library(shiny)
library(maps)
library(mapproj)
#LOAD DATA
counties <- readRDS("data/counties.rds")
#SOURCE HELPER FUNCTIONS
source("helpers.R")

# User interface ----
ui <- fluidPage(
  titlePanel("censusVis"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Create demographic maps with 
        information from the 2010 US Census."),
      
      selectInput("var", 
                  label = "Choose a variable to display",
                  choices = c("Percent White", "Percent Black",
                              "Percent Hispanic", "Percent Asian"),
                  selected = "Percent White"),
      
      sliderInput("range", 
                  label = "Range of interest:",
                  min = 0, max = 100, value = c(0, 100))
    ),
    
    mainPanel(plotOutput("map"))
  )
)

# Server logic ----
server <- function(input, output) {
  output$map <- renderPlot({
    data <- switch(input$var, 
                   "Percent White" = counties$white,
                   "Percent Black" = counties$black,
                   "Percent Hispanic" = counties$hispanic,
                   "Percent Asian" = counties$asian)
    color <- switch(input$var, 
                    "Percent White" = "darkgreen",
                    "Percent Black" = "black",
                    "Percent Hispanic" = "darkorange",
                    "Percent Asian" = "darkviolet")
    legend <- switch(input$var, 
                     "Percent White" = "% White",
                     "Percent Black" = "% Black",
                     "Percent Hispanic" = "% Hispanic",
                     "Percent Asian" = "% Asian")
    percent_map(data, color, legend, input$range[1], input$range[2])
  })
}

#A MORE CONCISE VERSION OF THE SERVER FUNCTION
# server <- function(input, output) {
#   output$map <- renderPlot({
#     args <- switch(input$var,
#                    "Percent White" = list(counties$white, "darkgreen", "% White"),
#                    "Percent Black" = list(counties$black, "black", "% Black"),
#                    "Percent Hispanic" = list(counties$hispanic, "darkorange", "% Hispanic"),
#                    "Percent Asian" = list(counties$asian, "darkviolet", "% Asian"))
#     
#     args$min <- input$range[1]
#     args$max <- input$range[2]
#     
#     do.call(percent_map, args)
#   })
# }

# Run app ----
shinyApp(ui, server)
