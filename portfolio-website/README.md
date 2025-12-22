# Portfolio Website

A modern, fully responsive portfolio website built with React, Vite, TailwindCSS, and Framer Motion. Showcasing projects in automotive engineering, robotics, and machine learning.

## 🚀 Features

- **Modern UI/UX**: Clean, professional design with automotive engineering theme
- **Fully Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Smooth Animations**: Subtle animations using Framer Motion
- **Dark Mode Ready**: Built with dark mode support (can be extended)
- **Multi-page Navigation**: React Router for seamless page transitions
- **Content Management**: Easy-to-edit JSON files for projects and profile data

## 📁 Project Structure

```
portfolio-website/
├── src/
│   ├── components/          # Reusable components
│   │   ├── Navbar.jsx       # Sticky navigation bar
│   │   ├── Footer.jsx       # Footer with social links
│   │   ├── ProjectCard.jsx  # Project card component
│   │   └── SectionTitle.jsx # Section title component
│   ├── pages/               # Page components
│   │   ├── Home.jsx         # Homepage
│   │   ├── About.jsx         # About page
│   │   ├── Projects.jsx     # Projects listing page
│   │   ├── Contact.jsx      # Contact page
│   │   ├── FYP.jsx          # FYP project detail page
│   │   ├── GoKart.jsx       # Go-Kart project detail page
│   │   ├── Cozmoclench.jsx  # Cozmoclench project detail page
│   │   └── Robotics.jsx     # Robotics portfolio page
│   ├── data/                # JSON data files
│   │   ├── profile.json     # Profile information
│   │   └── projects.json    # Projects data
│   ├── App.jsx              # Main app component with routing
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles with Tailwind
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── README.md
```

## 🛠️ Installation

1. **Navigate to the project directory:**
   ```bash
   cd portfolio-website
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   The website will be available at `http://localhost:5173` (or the port shown in terminal)

## 📝 Customization

### Updating Profile Information

Edit `src/data/profile.json` to update:
- Name, title, and bio
- Contact information (email, GitHub, LinkedIn)
- Experience and education
- Skills and expertise

### Adding/Editing Projects

Edit `src/data/projects.json` to:
- Add new projects
- Update project descriptions
- Modify tags and categories

### Styling

- **Colors**: Modify `tailwind.config.js` to change the color scheme
- **Fonts**: Update font imports in `index.html` and `tailwind.config.js`
- **Components**: Edit individual component files in `src/components/`

### Adding New Project Pages

1. Create a new page component in `src/pages/`
2. Add the project to `src/data/projects.json`
3. Add a route in `src/App.jsx`:
   ```jsx
   <Route path="/projects/your-project-id" element={<YourProjectPage />} />
   ```

## 🎨 Design Features

- **Automotive Theme**: Blue and cyan color scheme reflecting engineering/automotive vibes
- **Smooth Scrolling**: Sticky navbar with smooth scroll behavior
- **Hover Effects**: Interactive elements with hover states
- **Responsive Grid**: Adaptive layouts for all screen sizes
- **Professional Typography**: Inter font family for clean, modern look

## 📦 Build for Production

```bash
npm run build
```

The production build will be in the `dist/` directory.

## 🚀 Deploy

You can deploy this portfolio to:
- **Vercel**: Connect your GitHub repo and deploy automatically
- **Netlify**: Drag and drop the `dist` folder or connect via Git
- **GitHub Pages**: Use the `dist` folder contents
- **Any static hosting service**: Upload the `dist` folder

## 🛠️ Technologies Used

- **React 18**: UI library
- **Vite**: Build tool and dev server
- **TailwindCSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Framer Motion**: Animation library
- **Lucide React**: Icon library

## 📄 License

This project is open source and available for personal use.

## 🤝 Contributing

Feel free to fork this project and customize it for your own portfolio!

---

**Built with ❤️ using React, Vite, and TailwindCSS**
