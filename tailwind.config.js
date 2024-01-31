/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*", "./node_modules/flowbite/**/*.js"],
  theme: {
    extend: {
      colors: {
        "color-primary" : "#05A6C2",
        "color-bg" : "#F6FCFA",
        "color-primary-light" : "#62DEF1",
        "color-primary-dark" : "#010417",
        "color-secondary" : "#6D88FB",
        "color-secondary" : "#6D88FB",
        "color-gray" : "#333",
        "color-white" : "#fff",
        "color-blob" : "#a427df",
      },
      fontFamily: {
        poppins: ['Poppins', 'sans'],
        inter: ['Inter ', 'sans'],
      },
    },
    
    
    // container: {
    //     center: true,
    //     padding: {
    //       default: "20px",
    //       md: "50px"

    //     }
    // },
  },
  plugins: ["flowbite/plugin"],
}

